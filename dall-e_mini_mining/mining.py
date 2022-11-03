import os
import json
import base64
import requests
import threading

from time import sleep, time
from typing import List, Dict
from playsound import playsound


CIFAR10_CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

URL = "https://backend.craiyon.com/generate"
HEADERS = {
 "Accept": "application/json",
 "User-Agent": "Thunder Client",
 "Content-Type": "application/json" 
}

# Initializing class counters
class_counter = dict()
for _class in CIFAR10_CLASSES:
    class_counter[_class] = 0
    
request_counter = 0


def get_payload(prompt: str) -> Dict[str, str]:
    return json.dumps({
		"prompt": prompt
	})


def generate_images(prompt: str) -> List[str]:
    payload = get_payload(prompt)
    resp = requests.post(URL, headers=HEADERS, data=payload)
    images = []
    
    if resp.status_code == 200:
        images = resp.json()["images"]
    else:
        images = []
        playsound("off-hook-tone-3-beeps.mp3")
    
    global request_counter
    request_counter += 1
    print(f"Request {request_counter} for \"{prompt}\" : {resp.status_code}")

    return images


def save_image(b64_string_encoding: str, prompt: str) -> None:
    os.makedirs(f"fake_dataset_2/{prompt}", exist_ok=True)
    with open(f'fake_dataset_2/{prompt}/{class_counter[prompt]}.jpeg', 'wb') as fp:
        fp.write(base64.b64decode((b64_string_encoding)))
    class_counter[prompt] += 1
    
    
def mine_images(prompt) -> None:
    images = generate_images(prompt)
    for image in images:
        save_image(image, prompt)
        
    
def main():
    start_ts = time()
    threads = []
    for i in range(1000):
        for _class in CIFAR10_CLASSES:
            thread = threading.Thread(target=mine_images, args=[_class])
            thread.daemon = True
            thread.start()
            threads.append(thread)
            sleep(0.3)

    for thread in threads:
        thread.join()
        
    summary_msg = f"HTTP requests: {request_counter}"
    total = 0
    for _class in CIFAR10_CLASSES:
        summary_msg += f"\n  {_class}: {class_counter[_class]}"
        total += class_counter[_class]
    summary_msg += f"\nTotal images mined: {total}"
    summary_msg += f"\nTime elapsed: {round(time() - start_ts, 2)} s"
     
    print(summary_msg)
   

if __name__ == '__main__':
    main()
    