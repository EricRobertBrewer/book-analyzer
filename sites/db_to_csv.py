import pandas as pd
import os
import sqlite3

import folders


def write_to_csv(conn, table_name, csv_path, sort=None, force=False):
    if os.path.exists(csv_path):
        if force:
            os.remove(csv_path)
        else:
            print('CSV file already exists. Skipping.')
            return
    df = pd.read_sql_query('SELECT * FROM {};'.format(table_name), conn)
    if sort is not None:
        df.sort_values(sort, inplace=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')


def main(force=False):
    db_path = os.path.join(folders.CONTENT_PATH, 'contents.db')
    conn = sqlite3.connect(db_path)
    write_to_csv(conn, 'BookCaveBooks', folders.CONTENT_BOOKCAVE_BOOKS_CSV_PATH, sort='id', force=force)
    write_to_csv(conn, 'BookCaveBookRatings', folders.CONTENT_BOOKCAVE_BOOK_RATINGS_CSV_PATH, sort='book_id', force=force)
    write_to_csv(conn, 'BookCaveBookRatingLevels', folders.CONTENT_BOOKCAVE_BOOK_RATING_LEVELS_CSV_PATH, sort='book_id', force=force)
    conn.close()


if __name__ == '__main__':
    main(force=True)
