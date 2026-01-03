import os
import psycopg2
import psycopg2.extras
import requests
import json
import datetime
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global database connection
_db_connection = None


def get_db_connection():
    """Get database connection"""
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise Exception("DB_URL environment variable is required")
    return psycopg2.connect(db_url)

# def close_db_connection():
#     """Close the global database connection."""
#     global _db_connection
#     if _db_connection:
#         try:
#             _db_connection.close()
#             logger.info("Database connection closed")
#         except Exception as e:
#             logger.error(f"Error closing database connection: {e}")
#         finally:
#             _db_connection = None

def refresh_linear_token(refresh_token, viewer_name, client_id=None, client_secret=None):
    """Refresh Linear OAuth token using refresh token"""
    try:
        # Get client credentials from environment variables using viewer name if not provided
        if not client_id:
            client_id = os.getenv(f'{viewer_name}_CLIENT_ID')
        if not client_secret:
            client_secret = os.getenv(f'{viewer_name}_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print(f"❌ {viewer_name}_CLIENT_ID and {viewer_name}_CLIENT_SECRET are required for viewer {viewer_name}")
            return None
        
        print(f"🔄 Refreshing token for viewer {viewer_name}...")
        
        # Prepare the request
        url = 'https://api.linear.app/oauth/token'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        # Make the request
        response = requests.post(url, headers=headers, data=data)
        
        print(f"Token refresh response status for {viewer_name}: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            print(f"✅ Token refreshed successfully for viewer {viewer_name}!")
            return {
                'access_token': token_data.get('access_token'),
                'refresh_token': token_data.get('refresh_token'),
                'expires_in': token_data.get('expires_in'),
                'token_type': token_data.get('token_type')
            }
        else:
            print(f"❌ Token refresh failed for viewer {viewer_name}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error refreshing token for viewer {viewer_name}: {e}")
        return None

def refresh_viewer_tokens(conn, viewer_id, viewer_name, client_id=None, client_secret=None):
    """Refresh tokens for a specific viewer and update in database"""
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        print(f"🔄 Refreshing tokens for viewer_id: {viewer_id} ({viewer_name})")
        
        # Get current refresh token from database
        select_query = "SELECT refresh_token, viewer_name FROM cortex.linear_agents_tokens WHERE viewer_id = %s"
        cur.execute(select_query, (viewer_id,))
        record = cur.fetchone()
        
        if not record:
            print(f"❌ No record found for viewer_id: {viewer_id}")
            cur.close()
            return False
        
        current_refresh_token = record['refresh_token']
        db_viewer_name = record['viewer_name'] or viewer_name  # Use passed viewer_name if db doesn't have it
        
        if not current_refresh_token:
            print(f"❌ No refresh token found for viewer_id: {viewer_id}")
            cur.close()
            return False
        
        # Refresh the token using Linear API with viewer name
        new_tokens = refresh_linear_token(current_refresh_token, db_viewer_name, client_id, client_secret)
        
        if not new_tokens:
            print(f"❌ Failed to refresh tokens for viewer_id: {viewer_id} ({db_viewer_name})")
            cur.close()
            return False
        
        # Update database with new tokens
        success = update_viewer_tokens(
            conn=conn,
            viewer_id=viewer_id,
            access_token=new_tokens['access_token'],
            refresh_token=new_tokens['refresh_token'],
            expires_in=new_tokens['expires_in']
        )
        
        cur.close()
        
        if success:
            print(f"✅ Successfully refreshed and updated tokens for {db_viewer_name} ({viewer_id})")
            return True
        else:
            print(f"❌ Failed to update database for viewer_id: {viewer_id}")
            return False
            
    except Exception as e:
        print(f"❌ Error refreshing viewer tokens: {e}")
        return False

def refresh_all_tokens(client_id=None, client_secret=None):
    """Refresh tokens for all viewers in the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        print("🔄 Starting bulk token refresh for all viewers...")
        
        # Get all viewers with refresh tokens
        select_query = """
        SELECT viewer_id, viewer_name, refresh_token, expires_at 
        FROM cortex.linear_agents_tokens 
        WHERE refresh_token IS NOT NULL
        ORDER BY viewer_name
        """
        
        cur.execute(select_query)
        viewers = cur.fetchall()
        
        if not viewers:
            print("⚠️ No viewers with refresh tokens found")
            cur.close()
            return True
        
        print(f"Found {len(viewers)} viewers to refresh tokens for...")
        
        success_count = 0
        failed_count = 0
        
        for viewer in viewers:
            viewer_id = viewer['viewer_id']
            viewer_name = viewer['viewer_name'] or 'Unknown'
            
            print(f"\n🔄 Processing {viewer_name} ({viewer_id})...")
            
            # Check if token is close to expiry (refresh if expires within 24 hours)
            current_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
            expires_at = viewer['expires_at']
            
            # Convert expires_at to timestamp if it's a datetime object
            if expires_at:
                if isinstance(expires_at, datetime.datetime):
                    expires_at_timestamp = int(expires_at.timestamp())
                else:
                    expires_at_timestamp = int(expires_at)
                
                if expires_at_timestamp > (current_time + 86400):  # 24 hours
                    print(f"⏭️ Token for {viewer_name} still valid for more than 24 hours, skipping")
                    continue
            
            # Refresh tokens for this viewer with viewer_name parameter - pass connection
            if refresh_viewer_tokens(conn, viewer_id, viewer_name, client_id, client_secret):
                success_count += 1
            else:
                failed_count += 1
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
        
        cur.close()
        conn.close()
        print("Connection closed")
        print(f"\n📊 Bulk Token Refresh Summary:")
        print(f"   Total viewers processed: {len(viewers)}")
        print(f"   Successfully refreshed: {success_count}")
        print(f"   Failed to refresh: {failed_count}")
        
        return failed_count == 0
        
    except Exception as e:
        print(f"❌ Error during bulk token refresh: {e}")
        return False

def schedule_daily_refresh():
    """Schedule daily token refresh (basic implementation)"""
    print("🕐 Starting daily token refresh scheduler...")
    print("This will refresh tokens every 24 hours")
    print("Press Ctrl+C to stop")
    
    try:
        print(f"\n⏰ {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} - Starting scheduled token refresh")
        
        success = refresh_all_tokens()
        
        if success:
            print("✅ Scheduled token refresh completed successfully")
        else:
            print("⚠️ Some tokens failed to refresh during scheduled refresh")
        
        print("😴 Sleeping for 24 hours until next refresh...") 
            
    except KeyboardInterrupt:
        print("\n🛑 Daily refresh scheduler stopped by user")
    except Exception as e:
        print(f"❌ Error in daily refresh scheduler: {e}")

def update_viewer_tokens(conn, viewer_id, access_token, refresh_token, expires_in):
    """Update a specific viewer's tokens in the database"""
    try:
        cur = conn.cursor()
        
        print(f"Updating tokens for viewer_id: {viewer_id}")
        
        # Calculate expires_at as datetime object for timestamp with time zone column
        import datetime
        expires_at_datetime = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in)
        expires_at_timestamp = int(expires_at_datetime.timestamp())
        
        # Check if record exists
        check_query = "SELECT viewer_id FROM cortex.linear_agents_tokens WHERE viewer_id = %s"
        cur.execute(check_query, (viewer_id,))
        exists = cur.fetchone()
        
        if not exists:
            print(f"❌ No record found for viewer_id: {viewer_id}")
            cur.close()
            return False
        
        # Update the record with current timestamp
        current_time = datetime.datetime.now(datetime.timezone.utc)
        update_query = """
        UPDATE cortex.linear_agents_tokens 
        SET access_token = %s, refresh_token = %s, expires_at = %s, created_at = %s
        WHERE viewer_id = %s
        """
        
        cur.execute(update_query, (access_token, refresh_token, expires_at_datetime, current_time, viewer_id))
        
        if cur.rowcount > 0:
            conn.commit()
            print(f"✅ Successfully updated tokens for viewer_id: {viewer_id}")
            print(f"   New expires_at: {expires_at_timestamp} ({expires_at_datetime})")
            
            cur.close()
            return True
        else:
            print(f"⚠️ No rows were updated for viewer_id: {viewer_id}")
            cur.close()
            return False
        
    except Exception as e:
        print(f"❌ Error updating viewer tokens: {e}")
        return False

def main():
    """Main function to run the migration"""
    print("🚀 Starting Linear Agents Tokens Refresh")
    print("=" * 50)
    print("This script will refresh the tokens for the available linear agents")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv('DB_URL'):
        print("❌ SOURCE_DB_URL environment variable is not set!")
        print("Please set it to the connection string of your source database")
        return
    
    if not os.getenv('DATABASE_URL'):
        print("❌ DB_URL environment variable is not set!")
        print("Please set it to the connection string of your target database")
        return
    
    schedule_daily_refresh()
    print("\n🎉 Refreshing Access Token completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
