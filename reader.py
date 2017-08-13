from shove import Shove
import csv

mem_store = Shove()
root = Shove('file://shovestore')

print sorted(root.keys())
print root['2009-01-03']
print root['2016-01-03']
