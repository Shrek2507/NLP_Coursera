import re

string = input()

# Create a regex pattern...
pattern = r'[a-zA-Z]{5,}'

result = []

print(re.finditer(pattern, string))
