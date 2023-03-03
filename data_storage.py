import csv

#ime fajla gde upisujemo trajektoriju
coordinates_filename = "flies_path.csv"
interaction_filename = "interactions.csv"

#dodaj red
def input_trajectory(data):
    #otvori file za append
    with open(coordinates_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=range(len(data)))
        writer.writerow(data)

def input_interactions(data):
    # otvori file za append
    with open(interaction_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=range(len(data)))
        writer.writerow(data)



