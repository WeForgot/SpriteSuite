import os
import pickle
import random
import sqlite3
import time

from selenium import webdriver



class Scraper(object):
    Winner_Hint = {
        'Blue': 0,
        'Red': 1,
        'Crash': 2,
        'Skipped': 3,
        'Tie': 4,
        'Timeout': 5
    }
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
            Blue_Turns INTEGER DEFAULT 0,
            Red_0 INTEGER NOT NULL,
            Red_1 INTEGER,
            Red_2 INTEGER,
            Red_3 INTEGER,
            Red_Turns INTEGER DEFAULT 0,
            Outcome INTEGER NOT NULL,
            Session TEXT NOT NULL,
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
    
    def attempt_navigate(self, link, max_attempts=5, min_sleep=1.0, max_sleep=2.5):
        attempts = 0
        self.driver.get(link)
        while self.driver.title.startswith('500') and attempts < max_attempts:
            time.sleep(random.uniform(min_sleep, max_sleep))
            self.driver.get(link)
            attempts += 1
            if attempts == max_attempts:
                return False
        return True
    
    def get_character_ids(self, char_name_array):
        id_query = 'SELECT ID FROM Characters WHERE Name = ?'
        c = self.conn.cursor()
        to_return = []
        not_found = []
        # Using an array of character names, return an array of character IDs
        for name in char_name_array:
            if name is None:
                to_return.append(None)
                continue
            c.execute(id_query, (name,))
            fetched = c.fetchone()
            if fetched is None:
                not_found.append(name)
            else:
                idx = fetched[0]
                to_return.append(idx)
        if len(not_found) > 0:
            #print('Could not find {}, skipping'.format(', '.join(not_found)[:-2]))
            return None
        return to_return
    
    def get_ids(self, char_name_array):
        id_query = 'SELECT ID FROM Characters WHERE Name = ?'
        c = self.conn.cursor()
        went_bad = False
        for idx in range(len(char_name_array)):
            name = char_name_array[idx]
            if name is None:
                continue
            # Uncomment below if there is any reason to include these
            # From what I can tell they are either patched AI or replacements of the same name
            #cleaned_name = name.replace(' (dupe)','') if name.endswith('(dupe)') else name
            #cleaned_name = name.replace(' (old)','') if name.endswith('(old)') else name
            #c.execute(id_query, (cleaned_name,))
            c.execute(id_query, (name,))
            fetched = c.fetchone()
            if fetched is None:
                went_bad = True
                break
            char_name_array[idx] = fetched[0]
        return went_bad
    
    def parse_teams(self, blue_str, red_str):
        blue_turns = 0
        if ' / ' in blue_str:
            blue_arr = blue_str.split(' / ')
        elif ' ⇒ ' in blue_str:
            blue_arr = blue_str.split(' ⇒ ')
            blue_turns = 1
        else:
            blue_arr = [blue_str]
        blue_arr += [None] * (4 - len(blue_arr))
        
        red_turns = 0
        if ' / ' in red_str:
            red_arr = red_str.split(' / ')
        elif ' ⇒ ' in red_str:
            red_arr = red_str.split(' ⇒ ')
            red_turns = 1
        else:
            red_arr = [red_str]
        red_arr += [None] * (4 - len(red_arr))

        blue_went_bad = self.get_ids(blue_arr)
        red_went_bad = self.get_ids(red_arr)

        if blue_went_bad or red_went_bad:
            return None

        total_arr = blue_arr + [blue_turns] + red_arr + [red_turns]
        return total_arr

    def scrape_new_matches(self, done_matches, start_hint=0, min_sleep=1.0, max_sleep=2.5):   
        try:
            failures = open('failures.txt', 'a', encoding='utf-8')
            c = self.conn.cursor()
            index = start_hint
            template = 'https://mugen.spriteclub.tv/matches?startAt={}'
            id_query = 'SELECT ID FROM Characters WHERE Name = ?'
            query = 'INSERT INTO Matches (Blue_0, Blue_1, Blue_2, Blue_3, Blue_Turns, Red_0, Red_1, Red_2, Red_3, Red_Turns, Outcome, Session) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
            while True:
                if not self.attempt_navigate(template.format(index)):
                    self.conn.commit()
                    return index
                time.sleep(random.uniform(min_sleep, max_sleep))
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
                        continue
                    blue = match.find_element_by_css_selector('div.elem.matches-bluename').text
                    red = match.find_element_by_css_selector('div.elem.matches-redname').text

                    total_arr = self.parse_teams(blue, red)
                    if total_arr is None:
                        failures.write('Match ID:{}, Blue: {}, Red: {}\n'.format(match_id, blue, red))
                        failures.flush()
                        continue

                    winnerID = Scraper.Winner_Hint[match.find_element_by_css_selector('div.elem.matches-winner').text]

                    sessionType = match.find_element_by_css_selector('div.elem.matches-session > a').text.split('#')[0][:-1]

                    packed_variables = tuple(total_arr + [winnerID] + [sessionType])

                    c.execute(query, packed_variables)
                    done_matches.append(match_id)
                self.conn.commit()
                print('Completed matches {}->{}'.format(index, index+100))
                time.sleep(random.uniform(min_sleep, max_sleep))
                index += 100
        except KeyboardInterrupt as k:
            failures.write('------------------------------------------\n')
            failures.close()
            self.conn.commit()
            return
    
    def scrape_characters(self, division, link, done_characters, sleep_time=1, start_at=''):
        try:
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
                done_characters.append(charName)
                time.sleep(sleep_time)
            self.conn.commit()
            print(f'Completed the population of division {division}')
        except KeyboardInterrupt as k:
            self.conn.commit()
            return

if __name__ == '__main__':
    scraper = Scraper('sprite.db', headless=True)
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
    scraper.scrape_characters(-1, 'https://mugen.spriteclub.tv/characters?division=-1', previous_characters)
    scraper.scrape_characters(-2, 'https://mugen.spriteclub.tv/characters?division=-2', previous_characters)
    scraper.scrape_characters(-3, 'https://mugen.spriteclub.tv/characters?division=-3', previous_characters)
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
    scraper.shutdown()

