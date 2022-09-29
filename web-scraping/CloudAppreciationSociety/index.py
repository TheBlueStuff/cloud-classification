import json
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
import requests
import base64
import os
from bs4 import BeautifulSoup

def get_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument("--dns-prefetch-disable")
    chrome_options.add_argument('ignore-certificate-errors')
    chrome_options.add_argument('ignore-ssh-errors')
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
    #driver=webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(300)
    return driver
driver=get_driver()

def download_base64_image(data,filename):
    imgdata = base64.b64decode(data)
    with open(filename, 'wb') as f:
            f.write(imgdata)

def download_http_image(url,filename):
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        pass

def scraper():
    driver.get('https://cloudappreciationsociety.org/gallery/')
    filters = []
    filters_id = []    
    indexs = []
    count = 1
    nodes = driver.find_elements_by_xpath('//div[@class="cloud-filter--sections--section cloud-filter--main-cloud-types"]//ul[@class="cloud-filter"]/li/a')
    for node in nodes:
        node_text = node.get_attribute('innerText')
        node_id = node.get_attribute('rel')
        indexs.append(count)
        filters.append(node_text)
        filters_id.append(node_id)
        count += 1
    page=2
    output_data=[]
    for i in range(1, len(filters) + 1):
        img = 1
        img = "{0:0=4d}".format(img)
        isExist = os.path.exists("./images/"+str(indexs[i - 1])+'_' + filters[i - 1])
        if not isExist:
            os.makedirs("./images/"+str(indexs[i - 1])+'_' + filters[i - 1])
            print("The new directory is created!")
        count = 0
        offset = 0
        while True:
            url = "https://cloudappreciationsociety.org/wp-admin/admin-ajax.php"
            payload={'action': 'get_slideshow',
            'cats': str(filters_id[i - 1]),
            'offset': str(offset),
            'photographer': ''}
            files=[
            ]
            headers = {
            'Cookie': 'aelia_cs_selected_currency=GBP; aelia_customer_country=PK'
            }
            
            try:
                response = requests.request("POST", url, headers=headers, data=payload, files=files)
            except:
                if count == 10:
                    break
                else:
                    count += 1
                    continue
            json_data = json.loads(response.text)
            is_more = json_data['data']['more']
            if is_more == False:
                break
            else:
                offset=int(offset) + 16
                html_content = json_data['data']['content']
                content = '<html><head></head><body>' + html_content + '</body></html>'
                soup  =BeautifulSoup(content,'html.parser')
                nodes = soup.select('div.item')
                for j in range(1, len(nodes) + 1):
                    try:
                        file_name="./images/"+str(indexs[i - 1])+'_' + filters[i - 1]+'/'+str(indexs[i - 1])+'_' + filters[i - 1]+'_'+str(img)+'.jpg'
                        image_url = soup.select('div:nth-child('+str(j)+')>article img')[0]['src']
                        title = soup.select('div:nth-child('+str(j)+')>article h2')[0].text.replace('\n','').replace('\t','')
                        download_http_image(image_url, file_name)
                        print('Class '+ filters[i - 1] + " image "+ img + " Downloaded")
                        additional = []
                        tags = soup.select('div:nth-child('+str(j)+')>article span.cloud-filter>span>a')
                        for tag in tags:
                            tag_text = tag.text
                            if tag_text == filters[i - 1]:
                                continue
                            else:
                                try:
                                    tag_index = indexs[filters.index(tag_text)]
                                    additional.append(tag_index)
                                except:
                                    pass
                        
                        data_dic={
                            "path":file_name,
                            "image_title":title,
                            "class":filters[i - 1],
                            "id_class":indexs[i - 1],
                            "additional":additional
                        }
                        output_data.append(data_dic)
                        img = int(img) + 1
                        img = "{0:0=4d}".format(img)
                    except:
                        pass
                  
                
            
    with open('output_data.json', 'w') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)
        print('Data Saved to JSON...')
    print('Scraper Ended...')

scraper()