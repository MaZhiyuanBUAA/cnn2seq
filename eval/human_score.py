#human labeling system
import json
import sqlite3
#results.db for saving results

class labelling_system():
  def __init__():
    self.conn_data = sqlite3.connect("results.db")
    self.c_data = conn_data.cursor()
    self.conn_user = sqlite3.connect("user.db")
    self.c_user = conn_user.cursor()
    self.user_index = 0
  def create_tables(self):
    #create tables
    self.c_data.execute('''CREATE TABLE qa_pairs (id int primary key, query text, resp0 text, resp1 text, scores text)''')
    self.c_user.execute('''CREATE TABLE user_info (id int primary key, user_name text, labelling_logs text)''')
    self.conn_data.commit()
    self.conn_user.commit()

  def create_db(self):
    f = open('../data_bpe/test1k.query')
    querys = f.readlines()
    f.close()
    f = open('../results/seq2seq_4layers.txt')
    resp0s = f.readlines()
    f.close()
    f = open('../results/inputs_cnn.txt')
    resp1s = f.readlines()
    f.close()
    insert_items = zip(range(1,len(querys)+1),querys,resp0s,resp1s)
    self.c_data.executemany('INSERT INTO qa_pairs VALUES (?,?,?,?)',insert_items)
    self.conn_data.commit()

  def get_user(self,user_name):
    self.c_user.execute("SELECT * FROM user_info WHERE user_info.user_name=?",(user_name,))
    #print(c_user.fetchall())
    if not self.c_user.fetchall():
      self.c_user.execute('INSERT INTO user_info (id,user_name,labelling_logs) VALUES (?,?,?)',(self.user_index,user_name,json.dumps([])))
      self.user_index += 1
      self.conn_user.commit()
      print('user %s has been created'%user_name)
      return []
    c_user.execute("SELECT * FROM user_info WHERE user_info.user_name=?",(user_name,))
    user_info = c_user.fetchall()[0]
    return json.loads(user_info[2])
   
  def labelling(self):
    print('user_name:')
    user_name = raw_input().strip()
    user_logs = self.get_user(user_name)
    
    

if __name__ == '__main__':
  #create_tables()
  create_user('wyli')
  
