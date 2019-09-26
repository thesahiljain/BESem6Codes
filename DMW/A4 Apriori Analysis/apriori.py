from collections import Counter
# Get the user input
transactions = list()
print("Enter all transactions")
while True:
	t = input().split()
	if len(t) is 0:
		break
	transactions.append(t)
min_support = int(input("Enter minimum support : "))
print()

# Create initial table
table_set = set([t for transaction in transactions for t in transaction])
table_set = [frozenset([t]) for t in table_set]

# Method to remove min_support
def filter(table, min_support=min_support):
	return dict((item, table[item]) for item in table if table[item] >= min_support)

def count_dict(item_list, transactions=transactions):
	new_dict = dict()
	for item in item_list:
		new_dict[frozenset(item)] = 0
		for t in transactions:
			if item.issubset(set(t)):
				new_dict[frozenset(item)] += 1
	return new_dict

def join(item_list, k):
	table = [frozenset(set(x) | set(y)) for x in item_list for y in item_list if len(x | y) is k]
	return list(set(table))

def show_table(table):
	for key in table:
		print(set(key), "\t", table[key])
	print()

def support(x):
	count = 0
	for transaction in transactions:
		if set(x).issubset(set(transaction)):
			count += 1
	return count

# Afterwards
table = filter(count_dict(table_set))
print("Item set size: ", 1)
show_table(table)
k = 1
while True:

	new_table = filter(count_dict(join(table, k+1)))
	if(len(new_table)) is 0:
		break
	table = new_table
	k += 1
	print("Item set size: ", k)
	show_table(table)

# Rule generation
rule_items = [list(t) for t in table]

for rule in rule_items:
	for rhs in rule:
		lhs = [i for i in rule if i is not rhs]
		print("Rule : ", lhs, " -> ", rhs)
		print("Support : ", float(support(rule))/len(transactions)*100,"%")
		print("Confidence : ", float(support(rule))/support(lhs)*100,"%")
		print()

"""
Enter all transactions
m o n k e y
d o n k e y
m u c k y
m a k e
c o o k i e

Enter minimum support : 3

Item set size:  1
{'m'} 	 3
{'e'} 	 4
{'y'} 	 3
{'k'} 	 5
{'o'} 	 3

Item set size:  2
{'y', 'k'} 	 3
{'e', 'o'} 	 3
{'k', 'o'} 	 3
{'e', 'k'} 	 4
{'m', 'k'} 	 3

Item set size:  3
{'e', 'k', 'o'} 	 3

Rule :  ['k', 'o']  ->  e
Support :  60.0 %
Confidence :  100.0 %

Rule :  ['e', 'o']  ->  k
Support :  60.0 %
Confidence :  100.0 %

Rule :  ['e', 'k']  ->  o
Support :  60.0 %
Confidence :  75.0 %

"""