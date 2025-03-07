
# Function to initialize the database
def initialize_db():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            user_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (user_id, key)
        )
    ''')
    conn.commit()
    conn.close()

@tool
def add_key_value(key, value, user_id: Annotated[str, InjectedState("session")]):
    """Function to add or update a key-value pair"""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_data (user_id, key, value)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, key) 
        DO UPDATE SET value = excluded.value
    ''', (user_id, key, value))
    conn.commit()
    conn.close()

@tool
def get_key_values(user_id: Annotated[str, InjectedState("session")]):
    """Function to retrieve all stored key-value pairs"""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT key, value FROM user_data WHERE user_id = ?
    ''', (user_id,))
    results = cursor.fetchall()
    conn.close()
    return {key: value for key, value in results}