import requests, os, json, tqdm

if not os.path.exists("from_web"):
    os.makedirs("from_web")

with open("resources/url_to_filename.json") as fp:
    url_to_filename = json.load(fp)

    for url, name in tqdm.tqdm(url_to_filename, ncols=70):
        response = requests.get(url)

        if response.status_code == 200:
            with open(name, 'wb') as file:
                file.write(response.content)
        else:
            raise OSError("Failed to download %s. Please check your network and proxy settings first."%url)