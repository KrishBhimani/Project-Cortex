import os
import datetime
import asyncio
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from agno.team import TeamRunEvent
from agno.run.agent import RunEvent
from dotenv import load_dotenv
import requests
import json
from db.linear_tokens import schedule_daily_refresh
from core.context import AgentContext
from agents import AgentRegistry
import psycopg2
import psycopg2.extras

app = FastAPI()

# Load environment variables
load_dotenv()


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


@app.get('/callback')
async def callback(code: str = None, state: str = None):
    """
    GET endpoint to handle callback with code and state query parameters
    """
    print("=== CALLBACK ENDPOINT CALLED ===")
    
    print(f"Received code: {code}")
    print(f"Received state: {state}")
    
    if not code:
        raise HTTPException(status_code=400, detail={
            'error': 'Code parameter is required',
            'message': 'This endpoint expects to be called by Linear OAuth with a code parameter',
            'example': '/callback?code=your_oauth_code&state=your_state'
        })
    
    # Get client credentials from environment variables
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail={'error': 'CLIENT_ID and CLIENT_SECRET environment variables are required'})
    
    # Step 1: Exchange authorization code for access token
    token_url = 'https://api.linear.app/oauth/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    # Prepare form data
    data = {
        'code': code,
        'redirect_uri': 'https://a5f085fdd823.ngrok-free.app/callback',
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'authorization_code'
    }

    try:
        # Make POST request to Linear API for token exchange (offload blocking call)
        token_response = await asyncio.to_thread(
            requests.post, token_url, headers=headers, data=data
        )
        
        print(f"Token response status: {token_response.status_code}")
        print(f"Token response headers: {dict(token_response.headers)}")
        print(f"Token response text: {token_response.text}")
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail={
                'error': 'Token exchange failed',
                'status_code': token_response.status_code,
                'response': token_response.text,
                'sent_data': data
            })
        
        token_data = token_response.json()
        
        # Extract token information
        access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in')
        refresh_token = token_data.get('refresh_token')
        print(expires_in)
        if not access_token:
            raise HTTPException(status_code=400, detail={'error': 'No access token received'})
        
        # Step 2: Make GraphQL request to get viewer information
        graphql_url = 'https://api.linear.app/graphql'
        graphql_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        graphql_query = {
            "query": "query Viewer { viewer { id name organization { id name } } }"
        }
        
        # Make GraphQL request (offload blocking call)
        graphql_response = await asyncio.to_thread(
            requests.post, graphql_url, headers=graphql_headers, json=graphql_query
        )
        
        print(f"GraphQL response status: {graphql_response.status_code}")
        print(f"GraphQL response text: {graphql_response.text}")
        
        if graphql_response.status_code != 200:
            raise HTTPException(status_code=400, detail={
                'error': 'GraphQL request failed',
                'status_code': graphql_response.status_code,
                'response': graphql_response.text
            })
        
        graphql_data = graphql_response.json()
        
        # Extract viewer ID and name
        viewer_data = graphql_data.get('data', {}).get('viewer', {})
        viewer_id = viewer_data.get('id')
        viewer_name = viewer_data.get('name')
        
        if not viewer_id:
            raise HTTPException(status_code=400, detail={
                'error': 'Could not retrieve viewer ID',
                'graphql_response': graphql_data
            })
        
        # Store the tokens and viewer information in database (offload blocking call)
        print("=== SAVING TO DATABASE ===")
        db_success = await asyncio.to_thread(
            save_oauth_data, viewer_id, viewer_name, access_token, refresh_token, expires_in
        )
        
        if not db_success:
            print("WARNING: Database save failed, but OAuth flow completed successfully")
            return {
                'message': 'OAuth App Installed successfully (database save failed)',
                'viewer_id': viewer_id,
                'warning': 'Token data could not be saved to database - check DB_URL and network connectivity'
            }
        
        print("=== SUCCESS ===")

        return {
            'message': 'OAuth App Installed successfully',
            'viewer_name': viewer_name
        }
        
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail={
            'error': 'Request failed',
            'details': str(e)
        })


session_cache = {}


async def process_webhook_background(
    agent_session,
    app_user_id,
    agent_session_id,
    issue_data,
    event_type,
    access_token,
    agent_name,
    comment_body
):
    """
    Background task to process webhook: run agent and create response activity.
    This runs after the webhook endpoint has already returned.
    """
    try:
        issue_id = issue_data.get('id')
        issue_title = issue_data.get('title')
        issue_description = issue_data.get('description')
        team_data = issue_data.get('team', {})
        team_id = team_data.get('id')
        team_name = team_data.get('name')
        
        extracted_data = {
            'type': event_type,
            'issueId': issue_id,
            'title': issue_title,
            'teamId': team_id,
            'team_name': team_name,
        }
        
        if event_type == 'Comment':
            extracted_data['body'] = comment_body

        try:
            LINEAR_GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"
            LINEAR_API_KEY = os.getenv('LINEAR_API_KEY')
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
            
            # Offload blocking HTTP call
            issue = await asyncio.to_thread(fetch_project_from_issue, extracted_data['issueId'])
            print(issue, "\n\n")
            
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
            
            created_at = datetime.datetime.now(datetime.timezone.utc)
            
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
                
                # Authentication
                access_token=access_token,
                
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
            agent_result = None
            execution_start = datetime.datetime.now(datetime.timezone.utc)
            
            try:
                print(f"\n=== RESOLVING AGENT: {agent_name} ===")
                agent = AgentRegistry.get(agent_name)
                print(f"Agent resolved: {agent.__class__.__name__}")
                
                # Run agent asynchronously (agent.run is already async)
                agent_result = await agent.run(agent_context)
                print(f"Agent result: success={agent_result.success}, status={agent_result.status}")
                answer_comment = agent_result.response
                
            except KeyError as e:
                print(f"Agent routing failed: {e}")
                answer_comment = f"Error: Unknown agent '{agent_name}'. Available: {AgentRegistry.available_agents()}"
            except Exception as e:
                print(f"Agent execution failed: {e}")
                answer_comment = f"I encountered an error while processing your request: {str(e)}"
            
            # === SAVE EXECUTION TO DATABASE ===
            execution_end = datetime.datetime.now(datetime.timezone.utc)
            execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)
            
            try:
                from db.assembler import context_assembler
                
                # Extract structured output if available (Strategist returns this)
                structured_output = None
                if agent_result and hasattr(agent_result, 'metadata') and agent_result.metadata:
                    structured_output = agent_result.metadata.get('structured_output')
                
                await context_assembler.save_execution(
                    agent_name=agent_name,
                    issue_id=issue_id,
                    trigger_type='comment' if event_type == 'Comment' else 'issue',
                    trigger_comment_id=None,  # TODO: pass comment ID when available
                    trigger_body=comment_body if event_type == 'Comment' else issue_title,
                    input_context={
                        'issue_id': issue_id,
                        'issue_identifier': agent_context.issue_identifier,
                        'issue_title': issue_title,
                        'trigger_type': agent_context.trigger_type,
                        'project_id': project_id,
                        'project_name': project_name,
                    },
                    success=agent_result.success if agent_result else False,
                    status=agent_result.status if agent_result else 'error',
                    response_text=answer_comment,
                    structured_output=structured_output,
                    execution_time_ms=execution_time_ms,
                    error_message=None if (agent_result and agent_result.success) else answer_comment,
                )
                print(f"=== EXECUTION LOGGED ({execution_time_ms}ms) ===")
            except Exception as log_error:
                print(f"Failed to log execution: {log_error}")
                
        except Exception as db_error:
            print(f"Database error in webhook: {str(db_error)}")
            answer_comment = "Database error occurred."
            
        print("\nExtracted Data:\n", extracted_data)
        # Create second agent activity (response)
        print("=== CREATING RESPONSE ACTIVITY ===")
        response_content = {
            "type": "response",
            "body": answer_comment
        }
        
        # Offload blocking HTTP call
        await asyncio.to_thread(create_agent_activity, access_token, agent_session_id, response_content)
        
        print("=== BACKGROUND WEBHOOK PROCESSING COMPLETE ===")
        
    except Exception as e:
        print(f"Error in background webhook processing: {e}")


@app.post('/webhook')
async def webhook(request: Request):
    try:
        # Get JSON data from request body
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail={'error': 'No JSON data received'})

        print("=== WEBHOOK RECEIVED ===")
        print("\n\n", data, "\n\n")
        
        # Check if this is an AgentSessionEvent
        if data.get('type') != 'AgentSessionEvent':
            return {'message': 'Not an AgentSessionEvent, ignoring'}
        
        # Extract agentSession data
        agent_session = data.get('agentSession', {})
        if not agent_session:
            raise HTTPException(status_code=400, detail={'error': 'No agentSession found in webhook data'})
        
        extracted_data = {}
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
            
            # Extract common data
            app_user_id = data.get('appUserId')
            agent_session_id = agent_session.get('id')
            issue_data = agent_session.get('issue', {})
            issue_id = issue_data.get('id')
            issue_title = issue_data.get('title')
            issue_description = issue_data.get('description')
            team_data = issue_data.get('team', {})
            team_id = team_data.get('id')
            team_name = team_data.get('name')

            # Offload blocking DB call
            def get_agent_info():
                conn = get_db_connection()
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                check_query = 'SELECT expires_at, viewer_name FROM cortex.linear_agents_tokens WHERE viewer_id = %s'
                cur.execute(check_query, (app_user_id,))
                record = cur.fetchone()
                cur.close()
                conn.close()
                return record
            
            record = await asyncio.to_thread(get_agent_info)
            agent_name = record['viewer_name']
            expiry_of_linear = record['expires_at']
            print(f"Agent Name: {agent_name}, Token Expiry: {expiry_of_linear}")
            
            created_at = datetime.datetime.now(datetime.timezone.utc)
            if created_at > expiry_of_linear:
                # Offload blocking token refresh
                await asyncio.to_thread(schedule_daily_refresh)
            
            print("=== GETTING ACCESS TOKEN ===")
            # Offload blocking DB call
            access_token = await asyncio.to_thread(get_access_token_by_viewer_id, app_user_id)
            
            if not access_token:
                raise HTTPException(status_code=404, detail={
                    'error': 'No access token found for app user',
                    'app_user_id': app_user_id,
                    'message': 'User needs to complete OAuth flow first'
                })
            
            # Create first agent activity (thought)
            print("=== CREATING THOUGHT ACTIVITY ===")
            thought_content = {
                "type": "thought",
                "body": f"I am working on it.",
                "ephemeral": True
            }
            
            # Offload blocking HTTP call
            await asyncio.to_thread(create_agent_activity, access_token, agent_session_id, thought_content)
            
            extracted_data['type'] = event_type
            extracted_data['issueId'] = issue_id
            extracted_data['title'] = issue_title
            extracted_data['teamId'] = team_id
            extracted_data['team_name'] = team_name
            
            # Get comment body if it's a comment event
            comment_body = None
            if event_type == 'Comment':
                comment_data = agent_session.get('comment', {})
                comment_body = comment_data.get('body')
                extracted_data['body'] = comment_body

            # Start background processing for agent execution
            # This allows the webhook to return immediately while agent runs
            asyncio.create_task(process_webhook_background(
                agent_session=agent_session,
                app_user_id=app_user_id,
                agent_session_id=agent_session_id,
                issue_data=issue_data,
                event_type=event_type,
                access_token=access_token,
                agent_name=agent_name,
                comment_body=comment_body
            ))
            
            return {
                'message': f'Webhook received - {event_type} type, processing in background',
                'extracted_data': extracted_data,
                'activities_created': {
                    'thought': True,
                    'response': 'pending (background)'
                }
            }
        
        return {
            'message': f'skipped the webhook with same id',
            'extracted_data': extracted_data,
            'activities_created': 'No Activities'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail={
            'error': 'Failed to process webhook',
            'details': str(e)
        })


# ═══════════════════════════════════════════════════════════════════════════
# SYNC WEBHOOK - Syncs Linear issues and comments to cortex database
# ═══════════════════════════════════════════════════════════════════════════

from db.syncer import issue_syncer
from dateutil import parser as date_parser


@app.post('/sync_webhook')
async def sync_webhook(request: Request):
    """
    Webhook for syncing Linear issues and comments to cortex database.
    
    Handles:
    - Issue events (create, update)
    - Comment events
    
    Configure in Linear webhook settings to receive:
    - Issues
    - Comments
    """
    try:
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail={'error': 'No JSON data received'})
        
        event_type = data.get('type')
        action = data.get('action')
        
        print(f"=== SYNC WEBHOOK: {event_type} ({action}) ===")
        
        # Handle Issue events
        if event_type == 'Issue':
            issue_data = data.get('data', {})
            
            # Extract issue fields
            issue_id = issue_data.get('id')
            if not issue_id:
                return {'message': 'No issue ID, skipping'}
            
            # Parse timestamps
            created_at = date_parser.parse(issue_data.get('createdAt')) if issue_data.get('createdAt') else datetime.datetime.now(datetime.timezone.utc)
            updated_at = date_parser.parse(issue_data.get('updatedAt')) if issue_data.get('updatedAt') else created_at
            
            # Extract team info
            team = issue_data.get('team', {}) or {}
            team_id = team.get('id') or issue_data.get('teamId')
            team_name = team.get('name')
            
            # Extract project info
            project = issue_data.get('project', {}) or {}
            project_id = project.get('id') if project else None
            project_name = project.get('name') if project else None
            
            # Extract labels
            labels_data = issue_data.get('labels', []) or []
            labels = [l.get('name') for l in labels_data if l.get('name')] if isinstance(labels_data, list) else []
            
            # Extract assignee
            assignee = issue_data.get('assignee', {}) or {}
            assignee_id = assignee.get('id') if assignee else None
            
            # Extract creator
            creator = issue_data.get('creator', {}) or {}
            creator_id = creator.get('id') if creator else None
            
            # Sync to database
            success = await issue_syncer.sync_issue(
                issue_id=issue_id,
                identifier=issue_data.get('identifier', ''),
                title=issue_data.get('title', ''),
                description=issue_data.get('description'),
                state=issue_data.get('state', {}).get('name') if issue_data.get('state') else None,
                priority=issue_data.get('priority'),
                project_id=project_id,
                project_name=project_name,
                team_id=team_id,
                team_name=team_name,
                labels=labels,
                assignee_id=assignee_id,
                creator_id=creator_id,
                created_at=created_at,
                updated_at=updated_at,
            )
            
            return {
                'message': f'Issue {action} synced',
                'issue_id': issue_id,
                'success': success
            }
        
        # Handle Comment events
        elif event_type == 'Comment':
            comment_data = data.get('data', {})
            
            comment_id = comment_data.get('id')
            issue_id = comment_data.get('issueId') or (comment_data.get('issue', {}) or {}).get('id')
            
            if not comment_id or not issue_id:
                return {'message': 'Missing comment or issue ID, skipping'}
            
            # Parse timestamp
            created_at = date_parser.parse(comment_data.get('createdAt')) if comment_data.get('createdAt') else datetime.datetime.now(datetime.timezone.utc)
            
            # Extract author
            user = comment_data.get('user', {}) or {}
            author_id = user.get('id')
            author_name = user.get('name')
            
            # Check if this is an agent response (you can customize this logic)
            is_agent_response = comment_data.get('botActor') is not None
            agent_name = comment_data.get('botActor', {}).get('name') if is_agent_response else None
            
            # Sync to database
            success = await issue_syncer.sync_comment(
                comment_id=comment_id,
                issue_id=issue_id,
                body=comment_data.get('body', ''),
                author_id=author_id,
                author_name=author_name,
                created_at=created_at,
                is_agent_response=is_agent_response,
                agent_name=agent_name,
            )
            
            return {
                'message': f'Comment {action} synced',
                'comment_id': comment_id,
                'success': success
            }
        
        # Ignore other event types
        else:
            print(f"Ignoring event type: {event_type}")
            return {'message': f'Event type {event_type} not handled'}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in sync webhook: {e}")
        raise HTTPException(status_code=500, detail={
            'error': 'Failed to process sync webhook',
            'details': str(e)
        })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)