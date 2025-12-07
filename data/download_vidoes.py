import os
from pytubefix import YouTube

def in_progress_func(stream, chunk, bytes_remaining):
    print("Downloading...")

def completed_func(stream, file_path):
    print("Download completed!")


def download_video(url, file_name=None, save_path="./raw_videos/raw_videos"):
    if url == None:
        print("Error: Need URL to download a Youtube Video")
        return
    if os.path.exists(save_path + "/" + file_name + ".mp4"):
        print("Video already downloaded and in a directory!")
        return 
    yt = YouTube(url,
                on_progress_callback=in_progress_func,
                on_complete_callback=completed_func,
                use_oauth=False, 
                allow_oauth_cache=True).streams.get_highest_resolution().download(save_path)

    if file_name:
        print(yt)
        os.rename(yt, save_path + "/" + file_name + ".mp4")
    return yt




if __name__ == "__main__":

    race_list = {
        "race1": "https://www.youtube.com/watch?v=daWr9xnkKS4", # 2025 British Grand Prix
        "race2": "https://www.youtube.com/watch?v=ZgfOrbKkMsA", # 2025 US Grand Prix
        "race3": "https://www.youtube.com/watch?v=_PxZh0MWk2w", # 2025 Brazil Grand Prix
        "race4": "https://www.youtube.com/watch?v=IHE4-8uoT5E", # 2024 Best overtakes
        "race5": "https://www.youtube.com/watch?v=bHm-LRpn7ow", # 2025 Mexico GP
        "race6": "https://www.youtube.com/watch?v=ajzQj7bjSWE", # 2025 Monaco GP
        "race7": "https://www.youtube.com/watch?v=93ZnZF_zWds", # 2025 Canadian GP
        "race8": "http://youtube.com/watch?v=Hml6MaRRkn8", # 2025 Chinese GP
        "race9": "https://www.youtube.com/watch?v=bFXLP487kXo&t=95s", # 2025 Bahrain GP
        "race10": "https://www.youtube.com/watch?v=ZI-HntdeVas", # 2025 Miami GP
        "race11": "https://www.youtube.com/watch?v=xkRXnrvFCY0", # 2025 Italian GP
        "race12": "https://www.youtube.com/watch?v=Li93iQDZQeg", # 2025 Saudi GP
        "race13": "https://www.youtube.com/watch?v=ATlMK7ln5Dc", # 2025 Spanish GP
    }
    
    race_list_2 = {
        "race14": "https://www.youtube.com/watch?v=JntKOmbMI08"  # 2025 azberjain
    }
    # for k,v in race_list_2.items():
    #     print(f"Downloading: {k}")
    #     yt = download_video(url=v, file_name=k)
    link = "https://www.youtube.com/watch?v=YJ0NYHONwts"
    yt = download_video(url=link, file_name="race19")