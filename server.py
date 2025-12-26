import os
import datetime
import asyncio
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, Response
from agno.team import TeamRunEvent
from agno.run.agent import RunEvent
from dotenv import load_dotenv
import requests
import json
from refresh_linear_tokens import schedule_daily_refresh
from agent_context import AgentContext
from agents import AgentRegistry
# from linear_mcp import LinearMCP
# from backendtest import XOBackend
import psycopg2
import psycopg2.extras
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)

def run_async_in_thread(async_func):
    """Run async function in a separate thread with its own event loop"""
    def wrapper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func())
        finally:
            loop.close()
    
    future = executor.submit(wrapper)
    return future.result()

def get_db_connection():
    """Get database connection"""
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise Exception("DB_URL environment variable is required")
    return psycopg2.connect(db_url)

def create_agent_activity(access_token, agent_session_id, content):
    """Create agent activity using Linear GraphQL API"""
    try:
        graphql_url = 'https://api.linear.app/graphql'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        mutation_data = {
            "query": "mutation AgentActivityCreate($input: AgentActivityCreateInput!){ agentActivityCreate(input:$input){ success } }",
            "variables": {
                "input": {
                    "agentSessionId": agent_session_id,
                    "content": content
                }
            }
        }
        
        print(f"Creating agent activity: {content}")
        response = requests.post(graphql_url, headers=headers, json=mutation_data)
        
        print(f"Agent activity response status: {response.status_code}")
        print(f"Agent activity response: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to create agent activity: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error creating agent activity: {e}")
        return None

def update_issue_status(issue_id, state_id):
    """Update issue status using Linear GraphQL API"""
    try:
        graphql_url = 'https://api.linear.app/graphql'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': os.getenv('LINEAR_API_KEY')
        }
        
        mutation_data = {
            "query": "mutation UpdateIssueStatus($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success issue { id identifier title state { id name } } } }",
            "variables": {
                "id": issue_id,
                "input": {
                    "stateId": state_id
                }
            }
        }
        
        print(f"Updating issue status - Issue ID: {issue_id}, State ID: {state_id}")
        response = requests.post(graphql_url, headers=headers, json=mutation_data)
        
        print(f"Issue update response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('data', {}).get('issueUpdate', {}).get('success'):
                issue_data = result['data']['issueUpdate']['issue']
                print(f"Successfully updated issue {issue_data['identifier']} to state: {issue_data['state']['name']}")
                return result
            else:
                print(f"Failed to update issue status: {result}")
                return None
        else:
            print(f"Failed to update issue status: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error updating issue status: {e}")
        return None

def save_oauth_data(viewer_id, viewer_name, access_token, refresh_token, expires_in):
    """Save OAuth data to linear_agents_tokens table"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cortex.linear_agents_tokens (
            created_at timestamp with time zone NOT NULL DEFAULT now(),
            viewer_id text NOT NULL,
            access_token text NULL,
            refresh_token text NULL,
            viewer_name text NULL,
            expires_at timestamp with time zone NULL,
            CONSTRAINT linear_agents_tokens_pkey PRIMARY KEY (viewer_id)
        );
        """
        cur.execute(create_table_query)
        
        # Calculate expires_at as Unix timestamp (bigint)
        created_at = datetime.datetime.now(datetime.timezone.utc)
        expires_at=created_at + datetime.timedelta(seconds=expires_in-899)
        
        print(f"Saving to database - viewer_id: {viewer_id}, viewer_name: {viewer_name}, expires_at: {expires_at}")
        
        # First check if record exists
        check_query = 'SELECT viewer_id FROM cortex.linear_agents_tokens WHERE viewer_id = %s'
        cur.execute(check_query, (viewer_id,))
        existing_record = cur.fetchone()
        
        if existing_record:
            # Update existing record
            update_query = """
            UPDATE cortex.linear_agents_tokens 
            SET access_token = %s, refresh_token = %s, expires_at = %s, viewer_name = %s
            WHERE viewer_id = %s
            """
            cur.execute(update_query, (access_token, refresh_token, expires_at, viewer_name, viewer_id))
            print(f"Updated existing record for viewer_id: {viewer_id}")
        else:
            # Insert new record
            insert_query = """
            INSERT INTO cortex.linear_agents_tokens (viewer_id, viewer_name, access_token, refresh_token, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(insert_query, (viewer_id, viewer_name, access_token, refresh_token, expires_at))
            print(f"Inserted new record for viewer_id: {viewer_id}")
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"OAuth data saved successfully to linear_agents_tokens for viewer_id: {viewer_id}")
        return True
        
    except Exception as e:
        print(f"Error saving OAuth data: {e}")
        return False

def get_access_token_by_viewer_id(viewer_id):
    """Get access token from database by viewer_id"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        query = 'SELECT access_token FROM cortex.linear_agents_tokens WHERE viewer_id = %s'
        cur.execute(query, (viewer_id,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            print(f"Found access token for viewer_id: {viewer_id}")
            return result[0]
        else:
            print(f"No access token found for viewer_id: {viewer_id}")
            return None
            
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

@app.route('/callback', methods=['GET'])
def callback():
    """
    GET endpoint to handle callback with code and state query parameters
    """
    print("=== CALLBACK ENDPOINT CALLED ===")
    
    # Extract query parameters
    code = request.args.get('code')
    state = request.args.get('state')
    
    print(f"Received code: {code}")
    print(f"Received state: {state}")
    
    if not code:
        return jsonify({
            'error': 'Code parameter is required',
            'message': 'This endpoint expects to be called by Linear OAuth with a code parameter',
            'example': '/callback?code=your_oauth_code&state=your_state',
            'received_params': dict(request.args)
        }), 400
    
    # Get client credentials from environment variables
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    
    
    if not client_id or not client_secret:
        return jsonify({'error': 'CLIENT_ID and CLIENT_SECRET environment variables are required'}), 500
    
    # Step 1: Exchange authorization code for access token
    token_url = 'https://api.linear.app/oauth/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    # Prepare form data
    data = {
        'code': code,
        'redirect_uri': 'https://5b7331a75f8d.ngrok-free.app/callback',
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'authorization_code'
    }

    
    try:
        # Make POST request to Linear API for token exchange
        token_response = requests.post(token_url, headers=headers, data=data)
        
        print(f"Token response status: {token_response.status_code}")
        print(f"Token response headers: {dict(token_response.headers)}")
        print(f"Token response text: {token_response.text}")
        
        if token_response.status_code != 200:
            return jsonify({
                'error': 'Token exchange failed',
                'status_code': token_response.status_code,
                'response': token_response.text,
                'sent_data': data
            }), 400
        
        token_data = token_response.json()
        
        # Extract token information
        access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in')
        refresh_token = token_data.get('refresh_token')
        print(expires_in)
        if not access_token:
            return jsonify({'error': 'No access token received'}), 400
        
        # Step 2: Make GraphQL request to get viewer information
        graphql_url = 'https://api.linear.app/graphql'
        graphql_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        graphql_query = {
            "query": "query Viewer { viewer { id name organization { id name } } }"
        }
        
        
        # Make GraphQL request
        graphql_response = requests.post(graphql_url, headers=graphql_headers, json=graphql_query)
        
        print(f"GraphQL response status: {graphql_response.status_code}")
        print(f"GraphQL response text: {graphql_response.text}")
        
        if graphql_response.status_code != 200:
            return jsonify({
                'error': 'GraphQL request failed',
                'status_code': graphql_response.status_code,
                'response': graphql_response.text
            }), 400
        
        graphql_data = graphql_response.json()
        
        # Extract viewer ID and name
        viewer_data = graphql_data.get('data', {}).get('viewer', {})
        viewer_id = viewer_data.get('id')
        viewer_name = viewer_data.get('name')
        
        
        if not viewer_id:
            return jsonify({
                'error': 'Could not retrieve viewer ID',
                'graphql_response': graphql_data
            }), 400
        
        # Store the tokens and viewer information in database
        print("=== SAVING TO DATABASE ===")
        db_success = save_oauth_data(viewer_id, viewer_name, access_token, refresh_token, expires_in)
        
        if not db_success:
            print("WARNING: Database save failed, but OAuth flow completed successfully")
            return jsonify({
                'message': 'OAuth App Installed successfully (database save failed)',
                'viewer_id': viewer_id,
                'warning': 'Token data could not be saved to database - check DB_URL and network connectivity'
            })
        
        print("=== SUCCESS ===")

        return jsonify({
            'message': 'OAuth App Installed successfully',
            'viewer_name': viewer_name
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Request failed',
            'details': str(e)
        }), 500

session_cache={}
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Get JSON data from request body
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        print("=== WEBHOOK RECEIVED ===")
        # print(f"Full webhook data: {data}")
        print("\n\n",data,"\n\n")
        # Check if this is an AgentSessionEvent
        if data.get('type') != 'AgentSessionEvent':
            return jsonify({'message': 'Not an AgentSessionEvent, ignoring'}), 200
        
        # Extract agentSession data
        agent_session = data.get('agentSession', {})
        if not agent_session:
            return jsonify({'error': 'No agentSession found in webhook data'}), 400
        extracted_data={}
        if agent_session.get('id') not in session_cache:
            now = datetime.datetime.now(datetime.timezone.utc)
            print("\ninside the if condition\n")
            # Clean up expired sessions first
            expired = [sid for sid, expiry in session_cache.items() if expiry < now]
            for sid in expired:
                print(f"Session {sid} expired and removed")
                del session_cache[sid]
            expires_at = now + datetime.timedelta(seconds=1800)
            session_cache[agent_session.get('id')] = expires_at

            # Determine type based on sourceMetadata
            source_metadata = agent_session.get('sourceMetadata')
            event_type = 'Comment' if source_metadata is not None else 'Issue'
            
            print(f"Event type determined: {event_type}")
            # print(f"Source metadata: {source_metadata}")
            
            # Extract common data
            app_user_id = data.get('appUserId')
            agent_session_id = agent_session.get('id')
            issue_data = agent_session.get('issue', {})
            issue_id = issue_data.get('id')
            issue_title = issue_data.get('title')
            issue_description = issue_data.get('description')
            team_data = issue_data.get('team', {})
            team_id = team_data.get('id')
            team_name=team_data.get('name')

            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            check_query = 'SELECT expires_at, viewer_name FROM cortex.linear_agents_tokens WHERE viewer_id = %s'
            cur.execute(check_query, (app_user_id,))
            record = cur.fetchone()
            agent_name=record['viewer_name']
            expiry_of_linear=record['expires_at']
            print(f"Agent Name: {agent_name}, Token Expiry: {expiry_of_linear}")
            cur.close()
            conn.close()
            created_at = datetime.datetime.now(datetime.timezone.utc)
            if created_at>expiry_of_linear:
                schedule_daily_refresh()
            print("=== GETTING ACCESS TOKEN ===")
            access_token = get_access_token_by_viewer_id(app_user_id)
            
            if not access_token:
                return jsonify({
                    'error': 'No access token found for app user',
                    'app_user_id': app_user_id,
                    'message': 'User needs to complete OAuth flow first'
                }), 404
            
            # Create first agent activity (thought)
            print("=== CREATING THOUGHT ACTIVITY ===")
            thought_content = {
                "type": "thought",
                "body": f"I am working on it.",
                "ephemeral": True
            }
            
            thought_result = create_agent_activity(access_token, agent_session_id, thought_content)
            
            
            extracted_data['type']=event_type
            extracted_data['issueId']=issue_id
            extracted_data['title'] = issue_title
            extracted_data['teamId'] =team_id
            extracted_data['team_name'] = team_name
            
            # Initialize comment_body to None
            answer_comment = ""
            data={}
            comment_body = None
            if event_type == 'Comment':
                comment_data = agent_session.get('comment', {})
                comment_body = comment_data.get('body')
                extracted_data['body'] = comment_body
                

            try:
                LINEAR_GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"
                LINEAR_API_KEY=os.getenv('LINEAR_API_KEY')
                GET_PROJECT_BY_ISSUE = """
                    query GetProjectByIssue($issueId: String!) {
                    issue(id: $issueId) {
                        id
                        identifier
                        project {
                        id
                        name
                        }
                        labels {
                        nodes {
                            id
                            name
                        }
                        }
                    }
                    }
                    """
                def fetch_project_from_issue(issue_id: str):
                    headers = {
                        "Authorization": f"{LINEAR_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "query": GET_PROJECT_BY_ISSUE,
                        "variables": {"issueId": issue_id}
                    }

                    resp = requests.post(LINEAR_GRAPHQL_ENDPOINT, headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if "errors" in data and data["errors"]:
                        raise RuntimeError(json.dumps(data["errors"], indent=2))

                    return data.get("data", {}).get("issue")
                
                issue = fetch_project_from_issue(extracted_data['issueId'])
                print(issue,"\n\n")
                mcp_config = None
                labels = issue["labels"]["nodes"]
                label_names = [l["name"] for l in labels]
                print(label_names)

                # Extract project info if available
                project_id = None
                project_name = None
                if issue['project'] is not None:
                    project_id = issue['project']['id']
                    project_name = issue['project']['name']
                    print(f"Project: {project_id} - {project_name}")
                
                # Extract URLs from description and comment
                def extract_urls(text: str) -> list:
                    if not text:
                        return []
                    url_pattern = r'https?://[^\s<>"\')\]]+'
                    return re.findall(url_pattern, text)
                
                extracted_urls = extract_urls(issue_description or '')
                if comment_body:
                    extracted_urls.extend(extract_urls(comment_body))
                
                # === BUILD AGENT CONTEXT ===
                agent_context = AgentContext(
                    # Identity
                    agent_name=agent_name,
                    run_id=agent_session_id,
                    
                    # Issue Scope
                    issue_id=issue_id,
                    issue_identifier=issue.get('identifier', ''),
                    issue_title=issue_title or '',
                    issue_description=issue_description,
                    issue_state=None,  # Not currently extracted
                    issue_labels=label_names,
                    
                    # Trigger
                    trigger_type='comment' if event_type == 'Comment' else 'issue',
                    trigger_body=comment_body,
                    
                    # Project Scope
                    project_id=project_id,
                    project_name=project_name,
                    
                    # Retrieved Context (to be populated by retrieval layer)
                    project_kb_snippets=[],
                    related_issues=[],
                    
                    # External References
                    urls=extracted_urls,
                    
                    # Execution Metadata
                    user_id=app_user_id,
                    created_at=created_at,
                )
                
                print("=== AGENT CONTEXT BUILT ===")
                print(f"Agent: {agent_context.agent_name}")
                print(f"Issue: {agent_context.issue_identifier} - {agent_context.issue_title}")
                print(f"Trigger: {agent_context.trigger_type}")
                print(f"Project: {agent_context.project_name}")
                print(f"Labels: {agent_context.issue_labels}")
                print(f"URLs: {agent_context.urls}")
                
                # === AGENT ROUTING ===
                answer_comment = "I encountered an error while processing your request."
                try:
                    print(f"\n=== RESOLVING AGENT: {agent_name} ===")
                    agent = AgentRegistry.get(agent_name)
                    print(f"Agent resolved: {agent.__class__.__name__}")
                    
                    # Run agent asynchronously
                    async def run_agent():
                        return await agent.run(agent_context)
                    
                    result = run_async_in_thread(run_agent)
                    print(f"Agent result: success={result.success}, status={result.status}")
                    answer_comment = result.response
                    
                except KeyError as e:
                    print(f"Agent routing failed: {e}")
                    answer_comment = f"Error: Unknown agent '{agent_name}'. Available: {AgentRegistry.available_agents()}"
                except Exception as e:
                    print(f"Agent execution failed: {e}")
                    answer_comment = f"I encountered an error while processing your request: {str(e)}"
                
            except Exception as db_error:
                print(f"Database error in webhook: {str(db_error)}")
                answer_comment = "Database error occurred."
                
            print("\nExtracted Data:\n",extracted_data)
            # Create second agent activity (response)
            print("=== CREATING RESPONSE ACTIVITY ===")
            response_content = {
                "type": "response",
                "body": answer_comment
            }
            
            response_result = create_agent_activity(access_token, agent_session_id, response_content)
            
            
            return jsonify({
                'message': f'Webhook processed successfully - {event_type} type',
                'extracted_data': extracted_data,
                'activities_created': {
                    'thought': thought_result is not None,
                    'response': response_result is not None
                }
            })
        return jsonify({
            'message': f'skipped the webhook with same id',
            'extracted_data': extracted_data,
            'activities_created': 'No Activities'
        })
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({
            'error': 'Failed to process webhook',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)