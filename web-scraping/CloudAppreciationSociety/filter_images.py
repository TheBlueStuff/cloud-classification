import os
import pandas as pd
import json

def main():
    file = open('output_data.json', 'r')
    images = json.load(file) 

    filtered_images = []
    for image in images:
        if os.path.exists(image['path']) and image['additional'] == []:
            filtered_images.append(image)

    with open('filtered_data.json', 'w') as jsonfile:
        json.dump(filtered_images, jsonfile, indent=4)
    

if __name__ == '__main__':
    main()
