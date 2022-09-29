import json
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
import base64
import os
def get_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('ignore-ssh-errors')
    chrome_options.add_argument("--dns-prefetch-disable")
    chrome_options.add_argument('ignore-certificate-errors')
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
    urllib.request.urlretrieve(url, filename)

def scraper():
    driver.get('https://cloudatlas.wmo.int/en/search-image-gallery.html')
    filters = []
    indexs = []
    count = 1
    nodes = driver.find_elements_by_xpath('//select[@id="cldg"]/option[position()>1]')
    for node in nodes:
        node_text = node.text
        indexs.append(count)
        filters.append(node_text)
        count += 1
    page = 2
    json_data=[]
    for i in range(1, len(filters) + 1):
        img = 1
        img = "{0:0=4d}".format(img)
        driver.find_element_by_xpath('//select[@id="cldg"]/option[text()="' + filters[i - 1] + '"]').click()
        time.sleep(1)
        driver.find_element_by_xpath('//button[@id="btn_search_1"]').click()
        time.sleep(3)
        isExist = os.path.exists("./images/"+str(indexs[i - 1]) + '_'+filters[i - 1])
        if not isExist:
            os.makedirs("./images/"+str(indexs[i - 1])+'_' + filters[i - 1])
            print("The new directory is created!")
        count = 0
        while True:
            nodes=driver.find_elements_by_xpath('//div[@class="image_result"]/div/a/img')
            for j in range(1, len(nodes) + 1):
                while True:
                    try:
                        try:
                            driver.execute_script('arguments[0].click()', driver.find_element_by_xpath('//button[@id="cboxClose"]'))
                            time.sleep(2)
                        except:
                            pass
                        try:
                            driver.execute_script('arguments[0].click()', driver.find_element_by_xpath('//div[@class="image_result"]/div[' + str(j) + ']/a/img')) # Expand the image tab
                        except:
                            if count == 5:
                                break
                            else: 
                                time.sleep(2)
                                count += 1
                        time.sleep(2)
                        file_name="./images/"+str(indexs[i - 1]) + '_' + filters[i - 1] + '/' + str(indexs[i - 1]) + '_' + filters[i - 1] + '_' + str(img) + '.jpg'
                        image_url=driver.find_element_by_xpath('//img[@id="main_img"]').get_attribute('src')
                        if 'data:image/jpeg;base64' in image_url:
                            download_base64_image(image_url, file_name)
                        else:
                            download_http_image(image_url,file_name)
                        print('Class ' + filters[i - 1] + " image "+img+" Downloaded")
                        data_dic={
                            "path":file_name,
                            "class":filters[i - 1],
                            "class_index":indexs[i - 1]
                        }
                        json_data.append(data_dic)
                        img = int(img) + 1
                        img = "{0:0=4d}".format(img)
                        break
                    except Exception as e:
                        pass
                 # Close Image
                try:
                    driver.execute_script('arguments[0].click()', driver.find_element_by_xpath('//button[@id="cboxClose"]'))
                    time.sleep(2)
                except:
                    pass
                
            try:
                driver.find_element_by_xpath('//ul[@class="page_control"]/li/button[text()="' + str(page) + '"]').click()
                time.sleep(2)
                page += 1
            except:
                page = 1
                break
    with open('output_data.json', 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
        print('Data Saved to JSON...')
    print('Scraper Ended...')

scraper()