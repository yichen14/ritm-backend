import requests

resp = requests.post("http://localhost:5000/addimg",
                     files={"file": open('./images/cat.jpg','rb')})
print(resp.json())