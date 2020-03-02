# Loading raw json data into python specific data 
import json

faculty_json= """
{
	"faculty1": {
		"Name": {
			"First Name": "Meenakshi",
			"Middle Name": null,
			"Last Name": "Tripathi"
		},
		"Gender": "Female",
		"Designation": "Associate Professor",
		"Photo": "http://www.mnit.ac.in/PortalProfile/images/faculty/mnitjas137.png",
		"Department": "Computer Science and Engineering",
		"Qualifications": ["Ph.D. MNIT", "M.Tech. from Banasthali", "B.E. from Jaipur"],
		"Research Areas": ["Secure Routing In Wireless Sensor Networks", "Wireless Sensor Networks",
			"Wireless Ad hoc Networks", "Opportunistic Networks", "Network Security",
			"Software Defined Networks"
		],
		"Contact Details": {
			"Phone No.": [1412529154, 1412713419],
			"Email Id": "mtripathi.cse@mnit.ac.in"
		}
	},

	"faculty2": {
		"Name": {
			"First Name": "Pilli",
			"Middle Name": "Emmanuel",
			"Last Name": "Shubhakar"
		},
		"Gender": "Male",
		"Designation": "Head of Department",
		"Photo": "http://www.mnit.ac.in/PortalProfile/images/faculty/mnitjas200.jpg",
		"Department": "Computer Science and Engineering",
		"Qualifications": ["Ph.D. from IIT Roorkee", "M.Tech. from BIT Ranchi", " B.Tech. from Nagarjuna University"],
		"Research Areas": ["Security", "Privacy and Forensics", "Computer Networks",
			"Cloud Computing", "Big Data", "Internet of Things"
		],
		"Contact Details": {
			"Phone No.": 1412713376,
			"Email Id": "espilli.cse@mnit.ac.in"
		}
	}
}
"""

print (type(faculty_json))

# Converts  JSON Data types to Python Data Types 
# loads: load string
my_data = json.loads(faculty_json)

print(type(my_data) )  # its a python dictionary  , it uses the table to convert 
print(my_data) 
print()
print(my_data["faculty2"])

print (my_data['faculty2']['Qualifications'][0])  # Qualifications is list so can takeindex  


# Converts Python Data types to JSON Data Types
new_json_string = json.dumps(my_data)

#print (type(new_json_string) )
#print (new_json_string) 

new_json_string = json.dumps(my_data, indent=2 ) # show info in easy way
print (new_json_string) 

#new_json_string = json.dumps(my_data, indent=2, sort_keys=True) # sort the keys
#print (new_json_string)


# Writing/Storing the JSON data in a File 
with open("faculty_data.json", "w") as write_file:
    #json.dump(new_json_string, write_file)
    json.dump(new_json_string, write_file, indent=2 )


# Reading from a JSON file
with open("faculty_data.json", "r") as read_file:
    jsondata= json.load(read_file)
    print(jsondata)



###################################################################################









