import sqlite3

def main(filename, dbname):
    conn = sqlite3.connect(dbname)
    with open(filename, 'r', encoding='utf-8') as f:
        for ldx, line in enumerate(f):
            split_line = line.split(',')
            red_name = split_line[0]
            blue_name = split_line[1]
            results = conn.execute('SELECT red.ID, blue.ID FROM Characters AS red, Characters AS blue WHERE red.Name = ? AND blue.Name = ?', (red_name, blue_name)).fetchone()
            if results is None:
                continue
            red_id, blue_id = results
            winner = red_id if int(split_line[2]) == 0 else blue_id
            mode = 'Matchmaking' if split_line[6] == 'm' else 'Tournament'
            conn.execute('INSERT INTO Matches (Red,Blue,Winner,Mode) VALUES (?,?,?,?)', (red_id, blue_id, winner, mode))
            print(f'Adding {red_name} ({red_id}) vs {blue_name} ({blue_id}) (Winner: {winner})')
        conn.commit()

if __name__ == '__main__':
    main('saltyRecordsM--2020-01-27-13.20.txt', 'sprite.db')