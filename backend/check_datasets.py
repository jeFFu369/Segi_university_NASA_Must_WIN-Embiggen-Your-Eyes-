import sqlite3

# Connect to the database
conn = sqlite3.connect('nasa_images.db')
cursor = conn.cursor()

# Query to get all datasets
cursor.execute("SELECT * FROM datasets")
datasets = cursor.fetchall()

print("Datasets in the database:")
print("-" * 50)
for dataset in datasets:
    print(f"ID: {dataset[0]}")
    print(f"Name: {dataset[1]}")
    print(f"Description: {dataset[2]}")
    print(f"Created At: {dataset[3]}")
    print(f"Updated At: {dataset[4]}")
    print(f"Type: {dataset[5]}")
    print(f"Source: {dataset[6]}")
    print("-" * 50)

# Count the datasets
cursor.execute("SELECT COUNT(*) FROM datasets")
count = cursor.fetchone()[0]
print(f"Total datasets: {count}")

conn.close()