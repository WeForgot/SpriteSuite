import os
import pickle
import sqlite3
import time

from selenium import webdriver



class MatchScraper(object):
    def __init__(self, db_name, driver_location=None, headless=False):
        from selenium.webdriver.chrome.options import Options
        
        driveLoc = os.path.join('..','utils','chromedriver') if driver_location is None else driver_location
        if headless:
            options = Options()
            options.add_argument('--headless')
            self.driver = webdriver.Chrome(driveLoc, options=options)
        else:
            self.driver = webdriver.Chrome(driveLoc)
        self.conn = sqlite3.connect(db_name)
    
    def shutdown(self):
        self.driver.quit()
        self.conn.close()
    
    def create_tables(self):
        #conn.execute('SELECT * FROM Characters'):
        char_t_com = '''CREATE TABLE IF NOT EXISTS Characters (
            ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Division TEXT NOT NULL,
            Life INTEGER NOT NULL,
            Power INTEGER NOT NULL,
            Attack INTEGER NOT NULL,
            Defense INTEGER NOT NULL
        );'''
        match_t_com = '''CREATE TABLE IF NOT EXISTS Matches (
            ID INTEGER PRIMARY KEY,
            Blue_0 INTEGER NOT NULL,
            Blue_1 INTEGER,
            Blue_2 INTEGER,
            Blue_3 INTEGER,
            Red_0 INTEGER NOT NULL,
            Red_1 INTEGER,
            Red_2 INTEGER,
            Red_3 INTEGER,
            Turns INTEGER,
            Winner INTEGER NOT NULL,
            Mode TEXT NOT NULL,
            FOREIGN KEY (Red_0) REFERENCES Characters(ID),
            FOREIGN KEY (Red_1) REFERENCES Characters(ID),
            FOREIGN KEY (Red_2) REFERENCES Characters(ID),
            FOREIGN KEY (Red_3) REFERENCES Characters(ID),
            FOREIGN KEY (Blue_0) REFERENCES Characters(ID),
            FOREIGN KEY (Blue_1) REFERENCES Characters(ID),
            FOREIGN KEY (Blue_2) REFERENCES Characters(ID),
            FOREIGN KEY (Blue_3) REFERENCES Characters(ID)
        );'''

        c = self.conn.cursor()
        c.execute(char_t_com)
        c.execute(match_t_com)
    
    def attempt_navigate(self, link, max_attempts=5, sleep=5):
        attempts = 0
        self.driver.get(link)
        while self.driver.title.startswith('500') and attempts < max_attempts:
            time.sleep(sleep)
            self.driver.get(link)
            attempts += 1
            if attempts == max_attempts:
                return False
        return True

    def scrape_sessions(self, start_count=0):
        # Tournament example single elim title: Tournament #22511 - Single Elimination 1v1 - 4th Division
        # Tournament example double elim title: Tournament #22496 - Double Elimination 1v1 - 3rd Division
        # Tournament example 2v1 title: Tournament #22478 - Double Elimination 2v1 - 1592 Rated
        # Tournament example team title: Tournament #22502 - Single Elimination 3-Turn - 1234 Rated

        # Title Selector: #info-header
        # Table Selector: #stat-list
        # Single Row Selector: 
        #   Blue: #stat-list > div:nth-child(1) > div.elem.session-bluename
        #   Red: #stat-list > div:nth-child(1) > div.elem.session-redname
        #   Winner: #stat-list > div:nth-child(1) > div.elem.session-winner (additionally a span, but the span class is dependent on the winner, aka their color)        
        c = self.conn.cursor()
        index = start_count
        template = 'https://mugen.spriteclub.tv/matches?startAt={}'
        id_query = 'SELECT ID FROM Characters WHERE Name = ?'
        insert_com = 'INSERT INTO Matches (Red_0, Red_1, Red_2, Red_3, Blue_0, Blue_1, Blue_2, Blue_3, Turns, Winner, Mode) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        while True:
            if not self.attempt_navigate(template.format(index)):
                return index
            time.sleep(1)
            try:
                matches = self.driver.find_elements_by_class_name('stat-elem')
            except Exception as e:
                print(e)
                return index
            for match in matches:
                blue_text = match.find_element_by_css_selector('div.elem.matches-bluename').text
                red_text = match.find_element_by_css_selector('div.elem.matches-redname').text
                red = [None, None, None, None]
                blue = [None, None, None, None]
                red_ids = [None, None, None, None]
                blue_ids = [None, None, None, None]
                turns = 0
                if ' ⇒ ' in red or ' ⇒ ' in blue:
                    turns = 1
                    blue_split = blue_text.split(' ⇒ ')
                    red_split = red_text.split(' ⇒ ')
                    blue[:len(blue_split)] = blue_split
                    red[:len(red_split)] = red_split
                elif ' / ' in blue or ' / ' in red:
                    blue_split = blue_text.split(' / ')
                    red_split = red_text.split(' / ')
                    blue[:len(blue_split)] = blue_split
                    red[:len(red_split)] = red_split
                else:
                    blue[0] = blue_text
                    red[0] = red_text
                winner = match.find_element_by_css_selector('div.elem.matches-winner').text
                if winner == 'Skipped' or winner == 'Tie' or winner == 'Crash' or winner == 'Timeout':
                    continue
                mode = match.find_element_by_css_selector('div.elem.matches-session > a').text.split(' ')[0]
                for idx in range(len(red)):
                    try:
                        if red[idx] is not None:
                            c.execute(id_query, (red[idx],))
                            red_ids[idx] = c.fetchone()
                    except Exception as e:
                        print('Failed on {}'.format(red[idx]))
                        print(e)
                        return
                    try:
                        if blue[idx] is not None:
                            c.execute(id_query, (blue[idx],))
                            blue_ids[idx] = c.fetchone()
                    except Exception as e:
                        print('Failed on {}'.format(blue[idx]))
                        print(e)
                        return
                #print('-------------------------')
                #print('Blue: {} ({})\nRed: {} ({})\nWinner: {}\nMode: {}'.format(blue, blueID, red, redID, winner, mode))
                #print('-------------------------')
                winnerID = 0 if winner == 'Blue' else 1
                c.execute(insert_com, (*red_ids, *blue_ids, turns, winnerID, mode))
            self.conn.commit()
            print('Completed matches {}->{}'.format(index, index+100))
            time.sleep(3)
            index += 100

    def scrape_new_matches(self, done_matches, start_hint=0, sleep=0):
        # Tournament example single elim title: Tournament #22511 - Single Elimination 1v1 - 4th Division
        # Tournament example double elim title: Tournament #22496 - Double Elimination 1v1 - 3rd Division
        # Tournament example 2v1 title: Tournament #22478 - Double Elimination 2v1 - 1592 Rated
        # Tournament example team title: Tournament #22502 - Single Elimination 3-Turn - 1234 Rated

        # Title Selector: #info-header
        # Table Selector: #stat-list
        # Single Row Selector: 
        #   Blue: #stat-list > div:nth-child(1) > div.elem.session-bluename
        #   Red: #stat-list > div:nth-child(1) > div.elem.session-redname
        #   Winner: #stat-list > div:nth-child(1) > div.elem.session-winner (additionally a span, but the span class is dependent on the winner, aka their color)        
        try:
            c = self.conn.cursor()
            index = start_hint
            template = 'https://mugen.spriteclub.tv/matches?startAt={}'
            id_query = 'SELECT red.ID, blue.ID FROM Characters AS red, Characters AS blue WHERE red.Name = ? AND blue.Name = ?'
            insert_com = 'INSERT INTO Matches (Red, Blue, Winner, Mode) VALUES (?,?,?,?)'
            while True:
                if not self.attempt_navigate(template.format(index)):
                    self.conn.commit()
                    return index
                time.sleep(sleep)
                try:
                    matches = self.driver.find_elements_by_class_name('stat-elem')
                except Exception as e:
                    print(e)
                    self.conn.commit()
                    return index
                if len(matches) == 0:
                    return index
                for match in matches:
                    match_id = int(match.find_element_by_css_selector('div.elem.matches-matchid.li-key > a').text)
                    if match_id in done_matches:
                        print('Skipping match {}'.format(match_id))
                        continue
                    blue = match.find_element_by_css_selector('div.elem.matches-bluename').text
                    red = match.find_element_by_css_selector('div.elem.matches-redname').text
                    if ' / ' in blue or ' / ' in red or ' ⇒ ' in blue or ' ⇒ ' in red:
                        continue
                    winner = match.find_element_by_css_selector('div.elem.matches-winner').text
                    if winner == 'Skipped' or winner == 'Tie' or winner == 'Crash' or winner == 'Timeout':
                        continue
                    mode = match.find_element_by_css_selector('div.elem.matches-session > a').text.split(' ')[0]
                    c.execute(id_query, (red, blue))
                    results = c.fetchone()
                    if results is None:
                        continue
                    redID, blueID = results
                    #print('-------------------------')
                    #print('Blue: {} ({})\nRed: {} ({})\nWinner: {}\nMode: {}'.format(blue, blueID, red, redID, winner, mode))
                    #print('-------------------------')
                    winnerID = redID if winner == 'Red' else blueID
                    c.execute(insert_com, (redID, blueID, winnerID, mode))
                    done_matches.append(match_id)
                self.conn.commit()
                print('Completed matches {}->{}'.format(index, index+100))
                time.sleep(sleep)
                index += 100
        except KeyboardInterrupt as k:
            self.conn.commit()
            return
    
    def scrape_select_matches(self, match_range, sleep=1):
        c = self.conn.cursor()
        index = 0
        template = 'https://mugen.spriteclub.tv/matches?startAt={}'
        id_query = 'SELECT red.ID, blue.ID FROM Characters AS red, Characters AS blue WHERE red.Name = ? AND blue.Name = ?'
        insert_com = 'INSERT INTO Matches (Red, Blue, Winner, Mode) VALUES (?,?,?,?)'
        while len(match_range) > 0:
            if not self.attempt_navigate(template.format(index)):
                return index
            time.sleep(1)
            try:
                matches = self.driver.find_elements_by_class_name('stat-elem')
            except Exception as e:
                print(e)
                return index
            for match in matches:
                match_ID = int(match.find_element_by_css_selector('div.elem.matches-matchid.li-key > a').text)
                if match_ID in match_range:
                    match_range.remove(match_ID)
                    blue = match.find_element_by_css_selector('div.elem.matches-bluename').text
                    red = match.find_element_by_css_selector('div.elem.matches-redname').text
                    if ' / ' in blue or ' / ' in red or ' ⇒ ' in blue or ' ⇒ ' in red:
                        continue
                    winner = match.find_element_by_css_selector('div.elem.matches-winner').text
                    if winner == 'Skipped' or winner == 'Tie' or winner == 'Crash' or winner == 'Timeout':
                        continue
                    mode = match.find_element_by_css_selector('div.elem.matches-session > a').text.split(' ')[0]
                    c.execute(id_query, (red, blue))
                    results = c.fetchone()
                    if results is None:
                        continue
                    redID, blueID = results
                    #print('-------------------------')
                    #print('Blue: {} ({})\nRed: {} ({})\nWinner: {}\nMode: {}'.format(blue, blueID, red, redID, winner, mode))
                    #print('-------------------------')
                    winnerID = redID if winner == 'Red' else blueID
                    c.execute(insert_com, (redID, blueID, winnerID, mode))
                    
            index += 100
            time.sleep(3)
            print('Skimed matches {}->{}. {} remaining'.format(index, index+100, len(match_range)))
        self.conn.commit()
            
            

    def scrape_characters(self, division, link, done_characters, sleep_time=1, start_at=''):
        c = self.conn.cursor()
        self.driver.get(link)
        characters = self.driver.find_elements_by_class_name('characters-name')
        hrefs = []
        hold_tight = False
        if start_at != '':
            hold_tight = True
        for character in characters:
            character_name = character.find_element_by_tag_name('a').text
            if character_name in done_characters:
                continue
            if hold_tight:
                if character_name == start_at:
                    hold_tight = False
                else:
                    print(f'Skipping {character_name}')
                continue
            hrefs.append(character.find_element_by_tag_name('a').get_attribute('href'))
        if len(hrefs) == 0:
            print(f'No new characters in division {division}. Returning')
            return
        for href in hrefs:
            self.driver.get(href)
            attempts = 0
            while '500' in self.driver.title and attempts < 5:
                print('Failed to get page, refreshing')
                time.sleep(5)
                self.driver.get(href)
                attempts += 1
                if attempts == 5:
                    self.conn.commit()
                    return
            charDiv = division
            charName = self.driver.find_element_by_css_selector('#info-header > span').text
            charLife = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(1)').text.strip()
            charPow = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(2)').text.strip()
            charAtk = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(3)').text.strip()
            charDef = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(4)').text.strip()
            print('-------------------------------------------------')
            print(f'Name: {charName}\nDivision: {charDiv}\nLife: {charLife}\nPower: {charPow}\nAttack: {charAtk}\nDefense: {charDef}')
            c.execute('''INSERT INTO Characters (Name,Division,Life,Power,Attack,Defense) VALUES
                (?,?,?,?,?,?)''', (charName, charDiv, int(charLife), int(charPow), int(charAtk), int(charDef)))
            print('-------------------------------------------------')
            time.sleep(sleep_time)
        self.conn.commit()
        print(f'Completed the population of division {division}')
    
    def scrape_new_characters(self, sleep_time=1):
        div_to_link = {
            1: 'https://spriteclub.tv/characters?division=5',
            2: 'https://spriteclub.tv/characters?division=4',
            3: 'https://spriteclub.tv/characters?division=3',
            4: 'https://spriteclub.tv/characters?division=2',
            5: 'https://spriteclub.tv/characters?division=1'
        }
        c = self.conn.cursor()
        for div in div_to_link:
            if not self.attempt_navigate(div_to_link[div]):
                return div_to_link[div]
            characters = self.driver.find_elements_by_class_name('characters-name')
            hrefs = []
            for character in characters:
                char_link = character.find_element_by_tag_name('a')
                href = char_link.get_attribute('href')
                name = char_link.text
                c.execute('SELECT COUNT(*) FROM Characters WHERE Name=?', (name,))
                exists = c.fetchone()[0]
                if exists > 0:
                    continue
                hrefs.append(href)
                print('{} not found'.format(name))
            for href in hrefs:
                if not self.attempt_navigate(href):
                    return href
                charDiv = div
                charName = self.driver.find_element_by_css_selector('#info-header > span').text
                charLife = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(1)').text.strip()
                charPow = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(2)').text.strip()
                charAtk = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(3)').text.strip()
                charDef = self.driver.find_element_by_css_selector('#content > div:nth-child(1) > div:nth-child(3) > div.character-info.flex-row > div:nth-child(4)').text.strip()
                print('-------------------------------------------------')
                print(f'Name: {charName}\nDivision: {charDiv}\nLife: {charLife}\nPower: {charPow}\nAttack: {charAtk}\nDefense: {charDef}')
                c.execute('''INSERT INTO Characters (Name,Division,Life,Power,Attack,Defense) VALUES
                    (?,?,?,?,?,?)''', (charName, charDiv, int(charLife), int(charPow), int(charAtk), int(charDef)))
                print('-------------------------------------------------')
                time.sleep(sleep_time)
        self.conn.commit()
        



if __name__ == '__main__':
    scraper = MatchScraper('sprite_new.db', headless=True)
    scraper.create_tables()
    if os.path.exists('previous_characters.pkl'):
        print('Loading previous characters')
        with open('previous_characters.pkl', 'rb') as f:
            previous_characters = pickle.load(f)
    else:
        previous_characters = []
    scraper.scrape_characters(1, 'https://spriteclub.tv/characters?division=5', previous_characters)
    scraper.scrape_characters(2, 'https://spriteclub.tv/characters?division=4', previous_characters)
    scraper.scrape_characters(3, 'https://spriteclub.tv/characters?division=3', previous_characters)
    scraper.scrape_characters(4, 'https://spriteclub.tv/characters?division=2', previous_characters)
    scraper.scrape_characters(5, 'https://spriteclub.tv/characters?division=1', previous_characters)
    with open('previous_characters.pkl', 'wb') as f:
        print('Dumping previous characters')
        pickle.dump(previous_characters, f)

    if os.path.exists('previous_matches.pkl'):
        print('Loading previous matches')
        with open('previous_matches.pkl', 'rb') as f:
            previous_matches = pickle.load(f)
    else:
        print('Making new matches')
        previous_matches = []
    scraper.scrape_new_matches(previous_matches)
    with open('previous_matches.pkl', 'wb') as f:
        print('Dumping previous matches')
        pickle.dump(previous_matches, f)
    
    #scraper.scrape_sessions()
    #scraper.scrape_new_characters()
    #scraper.scrape_select_matches(list(range(697429,696215,-1)))
    scraper.shutdown()
    #db = sqlite3.connect('test.db')
    #c = db.cursor()
    #c.execute('''SELECT name FROM sqlite_master''')
    #tables = c.fetchall()
    #db.close()
    #assert ('Characters',) in tables and ('Matches',) in tables
    #os.remove('test.db')
