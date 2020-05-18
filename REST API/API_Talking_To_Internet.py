"""
How the HTTP Protocol Works 
TCP/IP was invented in 1983 ( kahn )
world wide web was invented in 1989 ( tim bernes lee )

Communication over the Internet requires the HTTP protocol understanding 
HTTP is a Language to communicate over the internet 
HTTP = Hyper Text Transfer Protocol 
Set of rules that governs how two devices should communicate 
HTTP is a plain text protocol, that means messages sent are in plain text 
HTTP is a stateless protocol
No record of previous interaction and each interaction is processed only with
the information that comes with a that particular interaction 
Self Destructing notes which are destroyed after read

How do we converse with a family or friend over a phone
They follow a set of convention ( rules )
You Call
You say Hello
Once Person asks question 
Other response back the question asked 
Then Goodbye
Hang Up 

If two machine talk to one another they need to have more specific rules

Tim Bernes Lee proposed www ( world wide web ) they proposed the first language draft of HTTP (0.9)  in 1991

Introduce the Client and the Sever concept 

Introduce the concept of Request and Response concept 


Introduction to the tool telnet
	HTTP is a plain text protocol, that means messages sent are in plain text 
	Telnet is a tool for Connecting and communicating with a remote device 
	
	Imagine  Server is a house with a unique address and that IP Address 
	This house has a lot of tenants living and each with a specific door for its room
	All the tenants speak a different language ( Hindi, English, Japanese etc )
	HTTP (80) , SMTP (25) , FTP ( 21 ) 

	telnet httpbin.org 80 and press enter once
	GET / HTTP/1.1 and press enter once ( This is know as the  REQUEST line )
	Host:httpbin.org and press enter twice ( Uniform Resource Identifier )
	CTRL + ]
	telnet>quit
	

	Discuss about the response from the server ( HTML )

	telnet httpbin.org 80 
	GET /xml HTTP/1.1  ( This is know as the  REQUEST line )
	Host:httpbin.org and press enter twice   

	Discuss about the response from the server ( XML )

	telnet httpbin.org 80
	GET /forsk HTTP/1.1 
	Host:httpbin.org and press enter twice  


	Fine tuning your request by sending some additional data 

	telnet httpbin.org 80
	GET /get?firstname=Sylvester&language=English HTTP/1.1 
	Host:httpbin.org and press enter twice  
 
	Introduce the concept of Query String 
 
	telnet httpbin.org 80
	GET /get?company=Forsk&city=Jaipur HTTP/1.1 
	Host:httpbin.org and press enter twice  




Format of the HTTP Request 
	
1	Request-Line	
    GET|POST [uri] HTTP/[version]	
    
    GET /xml HTTP/1.1

	GET is for viewing any resource on the server
	POST are used for making a change or updating 

	
2   Header	
    [Header Name]: [Header Value]	
    Host:httpbin.org
	User-Agent:telnet								
	Accept-Language:en-US									

3	Blank Line
	
4	Request Body
    (optional, only for POST)
    ( also known as payload)


Format of the HTTP Response 

1	 Status-Line	
     HTTP/[version] [status code] [status message]	
     HTTP/1.1 200 OK

    100 means Informational Messages 
    200 means all OK ( success messages ) 
    204 means No Content. The server successfully processed the request and did not return any content.
    301 means ( Resouce requested is moved to different location from the server )
    400 means Bad Request. The server cannot process the request because the client-side errors (incorrect request format).
    401 means Unauthorized. Occurs when authentication was failed, due to incorrect credentials or even their absence.
    404 means not found ( Something went wrong in client request ) 
    500 means Server Errors ( Something went working on server side while processing the request ) 
    

2	Headers		
    [Header Name]: [Header Value]	
    Server: nginx
	Date: 									
	Content-Type: application/xml										

3	 Blank Line

4	Response Body	[payload]					XML or HTML						



How to send data using the POST method of HTTP protocol 
	
	telnet httpbin.org 80
	POST /post HTTP/1.1 
	Host: httpbin.org
	Content-Length: 32 and press enter twice for blank line
	
	firstname=Chris&language=English
     	and press enter to send the command 

"""


"""

REST API Basics

REST is another layer on top of HTTP
When u r building a website / App you are building an UI for Apps logic and data model
API is to create a programatic interface or a code UI for same Logic and Data Model
All communication is done through HTTP
API = Application Programming Interface
REST = Representational State Transfer Protocol
Give the story of the Ola / Uber App Interfaces 


/api/v1/games?order=desc&sort=points

this would give back the games in descending order and sorted by points

Header information in request is a very important piece of information 

Request Headers
Accept 
specifies the file format the requester wants
Accept-Language
specifies the human readable language like English , Spanish or Russian
Cache-Control
specifies wether the response can be generated from a cache


Response Header 
Content-Type: text/javascript
Last-Modified: Tue, 15 Nov 2012 12:45:26 GMT
Expires: Thu, 01 Dec 2012 16:00:00 GMT
Status: 200 OK

Explain all the status codes 1xx, 2xx, 3xx, 4xx, 5xx


Endpoint = Specific URI 
Nouns and Verbs ( actions ) needs to be focused 
 
Endpoint points to a single record or a collection of records

/api/v1/games   ( represents a collection of games )
/api/v1/games/1234 ( represents a single games )
/api/v1/1234/games ( this is a wrong approach )

/api/v1/player
/api/v1/players/567


HTTP Methods  used in REST API
GET
is used for fetching either a collection of resources or a single resource.
POST
is used to add a a new resource to a collection
We would POST to /player or /players to crate a new player or game
PUT
is used when we want to update a record. We would not PUT on collection 
DELETE
is used for sending a DELETE request to a detail record, 
a URL for a single record should delete just that record. 
Sending DELETE to an entire collection would delete the whole collection 
but thats usually not implemented with good reason


Rate Limiting Concept of API
 Helps in DDOS 


API Authentication using API Token concept 
Pair A and Pair B keys are being used


"""




"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
READ FROM HERE, DONT READ THE TOP PART, YOU WOULD GET CONFUSED
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Introduce to the concept of Web Services ( REST API ) using openweathermap.org


http://api.openweathermap.org/data/2.5/weather?q=Jaipur

http://api.openweathermap.org/data/2.5/weather?q=Jaipur&appid=e9185b28e9969fb7a300801eb026de9c


# if you hit the URL in the browser and try to visualise the JSON, you will come to see that it has 12 items in the object 
# copy and paste this on json lint and visualise 

# coord as Object
    {
     #   lon as Number
     #   lat as Number
    }
# weather as List
    [ 
     {
         #  id as Number
         #  main as String
         #  description as String
         #  icon as String
     }
     ]
# base as String
# main as Object 
    {
        # temp as Number
        # pressure as Number
        # humidity as Number
        # temp_min as Number
        # temp_max as Number
    }
# visibility as Number
# wind as Object
    {
        # speed as Number
        # deg as Number
    }
# clouds as Object
    {
     # all as Number
    }
# dt as Number
# sys as Object
    # type as Number 
    # id as Number
    # message as Number
    # country as String
    # sunrise as Number
    # sunset as Number
# id as Integer
# name as String
# cod as Number



how to parse the response of a HTTP request which is in JSON format.

http://jsonplaceholder.typicode.com/
http://httpbin.org 
https://finance.yahoo.com/webservice/v1/symbols/allcurrencies/quote?format=json
"""



# Get me the temperature of the city from the openweathermap.org using request library


import requests

city = input("Enter city name: ")

url1 = "http://api.openweathermap.org/data/2.5/weather"
url2 = "?q="+city
url3 = "&appid=e9185b28e9969fb7a300801eb026de9c"

url = url1 + url2 + url3
print (url)


response = requests.get(url)

jsondata = response.json()
print (jsondata['main']['temp'])
# Find the Sunset and sunrise also
jsondata["sys"]["sunrise"]

jsondata["sys"]["sunset"]


#Or

payload = {"q":"Jaipur", "appid":"e9185b28e9969fb7a300801eb026de9c"}
response = requests.get(url1,params=payload)


# Content in binary form
print (type(response.content))
response.content


# Since we know that the content type is json we can call the json() function to get the data 
# converted to python data type (dict)
jsondata = response.json()
# response has the original JSON String
# jsondata has the convert string in the python data type dictionary



print (type(jsondata))

# to convert to Degree Centegrade we need to minus 273.15 
print (jsondata["main"]["temp"])


# Extra Explanation of the Response 
print ("Status code: " + str(response.status_code))
print ("Headers : " + str(response.headers))
print ("Data : " + response.text)


# Reading the Headers
for key, value in response.headers.items():
    print (key, ":" ,value , "\n")
   
print ("Content type: " + response.headers['content-type'])

print ("Content or Response Body : " + str(response.content))


# Reading the JSON Data
print (jsondata)

print (jsondata.keys())
print (jsondata.values())

print (len(jsondata.items()))

for key, value in jsondata.items():
    print (key, ":" ,value , "\n")
   

# Converting EPOCH time to HUMAN readable time
# The Unix epoch is the time 00:00:00 on 1 January 1970.
    
import time
time.ctime(jsondata["sys"]["sunrise"])
    

import datetime
datetime.fromtimestamp(jsondata["sys"]["sunset"])    

#################################################################

# Sample code to POST data

import json
import requests

Host = "http://httpbin.org/post"

data = {"firstname":"dev","language":"English"}

headers = {"Content-Type":"application/json","Content-Length":len(data),"data":json.dumps(data)}

def post_method():
    response = requests.post(Host,data,headers)
    return response

print ( post_method().text )




def get_method():
    response = requests.get("http://httpbin.org/get?firstname=Rohit")
    return response

print (get_method().text)

############################################################################

"""
How to use the urllib library
"""

    
import urllib
dir(urllib)  # contains (request, response,error,parse)

from urllib import request
dir(request)
 
   
resp = request.urlopen("https://www.wikipedia.org")
    
resp.code
    
resp.length  # in bytes
    
resp.peek() # its a byte object and not a string object
    
data = resp.read()   # reads entire response
    
type(data)
len(data)
    
html = data.decode('UTF-8')
    
type(html)
    
"""
# Introduce URL Encoding for Query String
qs = "key1=" + "value1" + "&" + "key2=" + "value2"
    
from urllib import parse
params = {"key1" :"value1", "key2" : "value2"} # create a dictionary
qs = parse.urlencode(params)



import urllib
url = "https://maps.googleappis.com/map/api/geocode/json?address=95130"
urllib.parse.urlencode({"address":"95130"})


"""

url = "https://api.thingspeak.com/update?api_key=27XV6RD4PCPCM965&field1=\"Yogendra\"&field2=25&field3=100"

# Write the data

import requests
response = requests.get(url)


#Read the data
url = "https://api.thingspeak.com/channels/956570/feeds.json?api_key=JL325E317YAFPSFO&results=2"
response = requests.get(url)

response.text


# Create a new Code Challenge to POST data 

"""
Code Challenge
  Name: 
    JSON Parser
  Filename: 
    json.py
  Problem Statement:
    Get me the other details about the city
        Latitude and Longitude
        Weather Condition
        Wind Speed
        Sunset Rise and Set timing in human readable format
"""


"""
Code Challenge
  Name: 
    Currency Converter Convert  from USD to INR
  Filename: 
    currecncyconv.py
  Problem Statement:
    You need to fetch the current conversion prices from the JSON  
    using REST API
  Hint:
    http://free.currencyconverterapi.com/api/v5/convert?q=USD_INR&compact=ultra&apiKey=07ca862e0b339dd56245
    or
    Check with http://api.fixer.io/latest?base=USD&symbol=EUR
    
"""

"""
Code Challenge:
    
http://forsk.in/images/Forsk_logo_bw.png"

Download the image from the url above and store in your working diretory with the same
name as the image name.

Do not hardcode the name of the image

"""


"""
Code Challenge

Download the image from the URL and store in a file

https://imgs.xkcd.com/comics/python.png

"""

# Code Challenge

"""
Research the below wesbite and post some data on it
https://requestbin.com
"""

# Extra Reading
# https://realpython.com/python-requests/

"""
Code Challenge
  Name: 
    Next Bus
  Filename: 
    nextbus.py
  Problem Statement:
    Write a Python code to find when will the next bus come
    Try to make the program generalised so that is in not hard wired for bus route 22 and stop id 14787
    Also try if you can take the arguments from the command line
    Also try if you can run the program as a script from comand line
    
  Sample:
      Route 22 and Stop 14787
      Route 0 and Stop 5037
      
  Hint: 
    http://ctabustracker.com/bustime/map/getStopPredictions.jsp?route=22&stop=14787 
"""





