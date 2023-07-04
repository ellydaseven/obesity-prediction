# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:08:31 2023

@author: Hp
"""

import sqlite3
conn = sqlite3.connect("data.db",check_same_thread=False)
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS dataEntries(age INTEGER, height INTEGER, weight INTEGER, fastFood TEXT, carbonDrink TEXT, alcohol TEXT, exercise TEXT, sports TEXT, obesity TEXT)')
    
def add_data(age, height, weight, fastFood, carbonDrink, alcohol, exercise, sports, obesity):
    c.execute('INSERT INTO dataEntries(age, height, weight, fastFood, carbonDrink, alcohol, exercise, sports, obesity) VALUES (?,?,?,?,?,?,?,?,?)',(age, height, weight, fastFood, carbonDrink, alcohol, exercise, sports, obesity))
    conn.commit()

def view_data():
    c.execute('SELECT * FROM dataEntries')
    result = c.fetchall()
    return result
