from database_connect import DDSMDataset

#Initialize the object
dataset = DDSMDataset()

#Get the patient data for all four views
print(dataset.get_grouped_data(1))

#Get individual images stored in the database
print(dataset.get_grouped_data(1,single=True))

#Get the number of patients in the database
print(dataset.get_length())
