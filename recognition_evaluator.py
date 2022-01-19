import pandas as pd

student_results = pd.read_csv('Output.csv')
ground_truth = pd.read_csv('groundTruth.csv')

counter = 0
total = 15
correct = 0
incorrect = 0

while counter < total:
    student_plate = student_results['License plate'][counter]
    solution_plate = ground_truth['License plate'][counter]

    # Remove - and quotes
    student_plate = student_plate.replace('-', '').replace('\'','')
    solution_plate = solution_plate.replace('-', '')

    for index, char in enumerate(student_plate):
        if char == solution_plate[index]:
            correct += 1
    counter += 1

incorrect = (total * 6) - correct
result = correct / (total * 6)

# Show results
print("RESULTS:")
print("---------------------------------------")
print("INCORRECT: ", incorrect)
print("CORRECT: ", correct)
print("ACCURACY: ", correct, "/", (total * 6), "=", result)
