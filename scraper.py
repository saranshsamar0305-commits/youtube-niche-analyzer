import pandas as pd
from googleapiclient.discovery import build
import time

# TODO: Replace with your actual Google Cloud API Key
API_KEY = "AIzaSyBrckCbkP4Qg6bzQ3RxRQ_0X5EsvYZL_m4"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_data(query, max_results=50):
    print(f"Fetching data for: {query}...")
    videos = []
    
    try:
        # Search for videos
        request = youtube.search().list(
            q=query, part='snippet', type='video', maxResults=max_results
        )
        response = request.execute()
        
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            
            # Get video statistics (views, likes)
            stat_request = youtube.videos().list(
                part='statistics', id=video_id
            )
            stat_response = stat_request.execute()
            
            if stat_response.get('items'):
                stats = stat_response['items'][0]['statistics']
                view_count = int(stats.get('viewCount', 0))
                like_count = int(stats.get('likeCount', 0))
                
                # Synthetic RPM Proxy based on engagement
                engagement_rate = (like_count / view_count) if view_count > 0 else 0
                value_score = min(100, engagement_rate * 1000)
                
                videos.append({
                    'title': title,
                    'description': description,
                    'views': view_count,
                    'likes': like_count,
                    'value_score': value_score
                })
    except Exception as e:
        print(f"Error fetching data: {e}")
            
    return pd.DataFrame(videos)

if __name__ == "__main__":
    niches = ["AI hardware", "personal finance", "crypto trading", "productivity setup"]
    
    all_data = pd.DataFrame()
    for niche in niches:
        df = get_video_data(niche, max_results=50)
        all_data = pd.concat([all_data, df], ignore_index=True)
        time.sleep(1) # Be polite to the API
        
    all_data.to_csv("data/youtube_dataset.csv", index=False)
    print("Data saved to data/youtube_dataset.csv")