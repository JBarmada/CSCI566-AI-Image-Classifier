import concurrent.futures
import urllib.request

URLS = {
    'real': 'https://drive.usercontent.google.com/download?id=1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw&export=download&confirm=t',
    'fomm': 'https://huggingface.co/datasets/ManhQuangAI/df-40-test-full/resolve/b052e68b675c4c52b36439cfa1c88bdb49bb92b8/fomm.zip?download=true',
    'wav2lip': 'https://huggingface.co/datasets/ManhQuangAI/df-40-test-full/resolve/b052e68b675c4c52b36439cfa1c88bdb49bb92b8/wav2lip.zip?download=true',
    'simswap': 'https://huggingface.co/datasets/ej2/df40/resolve/e0d28c7b8ca85aa5a8102e1850e2c271fb23f24e/simswap.zip?download=true',
}


def probe(item):
    name, url = item
    try:
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023'})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(1024)
            return (name, response.status, response.headers.get('Content-Type'),
                    response.headers.get('Content-Range'), data[:80])
    except Exception as error:
        return name, str(error)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(probe, URLS.items()):
        print(result)
