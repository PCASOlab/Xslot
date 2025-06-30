import csv

# Define the path to your CSV file
csv_file_path = "/media/guiqiu/Installation/database/surgtoolloc2022_dataset/_release/training_data/labels.csv"

# Initialize an empty list to store the data from the CSV file
data = []

# Open the CSV file and read its contents
try:
    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Read the header row (if any)
        header = next(csvreader)

        # Read the remaining rows and append them to the 'data' list
        for row in csvreader:
            data.append(row)
except FileNotFoundError:
    print(f"File not found at path: {csv_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Now you have the data from the CSV file in the 'data' list
# You can manipulate or process the data as needed

# Example: Printing the first few rows
for row in data[:5]:
    print(row)





