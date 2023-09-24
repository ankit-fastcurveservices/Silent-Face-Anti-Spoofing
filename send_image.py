import requests
url = 'http://127.0.0.1:8080/liveness/classify'
my_img = {'image': open('images/sample/image_T1.jpg', 'rb')}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
print(r.json())
