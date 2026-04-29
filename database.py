import sqlite3

def get_athlete_historical_risk(athlete_id="user_01"):
    """
    Simulates a call to your PostgreSQL database.
    Fetches historical risk based on past injuries and biometric data.
    """
    # Create a temporary in-memory DB for the demo
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''CREATE TABLE athlete_history 
                      (id TEXT, prev_injuries INT, fatigue_index FLOAT)''')
    
    # Insert mock data
    cursor.execute("INSERT INTO athlete_history VALUES ('user_01', 2, 0.65)")
    
    # Query data
    cursor.execute("SELECT prev_injuries, fatigue_index FROM athlete_history WHERE id=?", (athlete_id,))
    row = cursor.fetchone()
    
    # Calculate a 'Static Risk Score' (0.0 - 1.0)
    # More injuries + high fatigue = higher starting risk
    static_risk = (row[0] * 0.2) + (row[1] * 0.3)
    return min(static_risk, 1.0)

if __name__ == "__main__":
    risk = get_athlete_historical_risk()
    print(f"Database Query Successful. Historical Risk Factor: {risk}")
