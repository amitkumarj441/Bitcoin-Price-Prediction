from shove import Shove
import csv

mem_store = Shove()
root = Shove('file://shovestore')


def parse(filename):
  print 'Parsing', filename
  n, e = filename.split('.')
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      date = row[0]
      if date == 'unixtime':
        names = row
        continue

      o = root[date] = root.get(date) or {}
      for index, name in enumerate(names):
        key = name if index is 0 else (n+'.'+name)
        o[key] = row[index]
      print date, len(o.keys())


parse('bitfinexUSD.csv')


root.sync()
