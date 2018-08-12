-- Intro To SQL for Data Science



-- Comments (single line)

/* Comment
	(multiple line) */









-- CHAPTER 1: SELECTING COLUMNS








-- Select the release_year column from the films table
SELECT release_year
from films;

-- Select multiple columns from films table
SELECT title, release_year
FROM films;

-- Select all columns from film table
SELECT *
FROM films;

-- Select only distinct values from a column
SELECT DISTINCT country
FROM films;

-- Count number of rows from reviews table
SELECT COUNT(*)
FROM reviews;

-- Count non-missing entries
SELECT COUNT(birthdate)
FROM people;

-- Count distinct values from birthdate column in people table
SELECT COUNT(DISTINCT birthdate)
FROM people;









-- CHAPTER 2: FILTERING ROWS








-- Filtering with WHERE
SELECT *
FROM films
WHERE release_year=2016

-- Count a filtered selection
SELECT COUNT(*)
FROM films
WHERE release_year<2000

-- Select multiple criteria based on filter 
SELECT title, release_year
FROM films
WHERE release_year>2000

-- Filtering based on text 
SELECT *
FROM films
WHERE language='French'

-- Select based on date 
SELECT name, birthdate
FROM people
WHERE birthdate = ('1974-11-11')

-- Select based on multiple criteria
SELECT title, release_year
FROM films
WHERE release_year<2000
AND language='Spanish'

-- Select based on multiple criteria
SELECT *
FROM films
WHERE release_year>2000
AND release_year<2010
AND language='Spanish'

-- Select by Where, and, or conditions
SELECT title, release_year
FROM films
WHERE (release_year>1989 AND release_year<2000)
AND (language = 'French' OR language = 'Spanish')
AND (gross > 2000000)

-- BETWEEN: Select based on numeric range using BETWEEN
SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 and 2000
AND budget>100000000
AND (language = 'Spanish' or language = 'French')

-- IN: Select based on IN clause, subsitutes complex where clause
SELECT title, certification
FROM films
WHERE certification IN ('NC-17', 'R')

-- ISNULL: select based on missing values
SELECT name, deathdate
FROM people
WHERE deathdate ISNULL

--ISNULL paired with COUNT
SELECT COUNT(*)
FROM films
WHERE language ISNULL

-- LIKE AND NOT LIKE
-- The % wildcard will match zero, one, or many characters in text. For example, 
-- the following query matches companies like 'Data', 'DataC' 'DataCamp', 'DataMind', and so on:
-- The _ wildcard will match a single character. For example, the following query matches companies like 'DataCamp', 'DataComp', and so on:
SELECT name 
FROM people
WHERE name LIKE 'B%'

name
B.J. Novak
Babak Najafi

SELECT name 
FROM people
WHERE name LIKE '_r%'

name
Ara Celi
Aramis Knight

SELECT name 
FROM people
WHERE name NOT LIKE 'A%'

name
50 Cent
Álex Angulo








--- Chapter 3: Aggregate Functions







-- Aggregation Functions: SUM, MIN, MAX, AVG

SELECT AVG(duration)
FROM films;

SELECT MIN(duration)
FROM films;


-- Aggregate functions with WHERE

SELECT SUM(gross)
FROM films
WHERE release_year >= 2000

SELECT AVG(gross)
FROM films
WHERE title LIKE 'A%'

SELECT MAX(gross)
FROM films
WHERE release_year BETWEEN 2000 AND 2012


-- It's AS simple AS aliasing

SELECT MAX(budget) AS max_budget,
       MAX(duration) AS max_duration
FROM films;

SELECT title, 
    (gross-budget) AS net_profit
FROM films;

SELECT title, 
    (duration/60.0) AS duration_hours
FROM films;

SELECT AVG(duration/60.0) AS avg_duration_hours
FROM films;


-- Even more aliasing

-- get the count(deathdate) and multiply by 100.0
-- then divide by count(*)

SELECT COUNT(deathdate) * 100.0 / COUNT(*) AS percentage_dead
FROM people

SELECT MAX(release_year) - MIN(release_year)
    AS difference
FROM films;







--- Chapter 4: Sorting, grouping and joins




-- ORDER BY

SELECT name
FROM people
ORDER BY name

SELECT name, birthdate
FROM people
ORDER BY birthdate

-- Order by single column

SELECT title
FROM films
WHERE release_year IN (2000, 2012)
ORDER BY release_year

SELECT *
FROM films
WHERE release_year != 2015
ORDER BY duration

SELECT title, gross
FROM films
WHERE title LIKE 'M%'


-- Sorting single columns (DESC)

SELECT imdb_score, film_id
FROM reviews
ORDER BY imdb_score DESC;

SELECT title, duration
FROM films
ORDER BY duration DESC;


--Sorting multiple columns

SELECT birthdate, name
FROM people
ORDER BY birthdate, name

SELECT certification, release_year, title
FROM films
ORDER BY certification, release_year



-- GROUP BY practice

SELECT release_year, COUNT(*)
FROM films
GROUP BY release_year

SELECT release_year, AVG(duration)
FROM films
GROUP BY release_year

SELECT release_year, MAX(budget)
FROM films
GROUP BY release_year

SELECT release_year, MIN(gross)
FROM films
GROUP BY release_year

SELECT language, SUM(gross)
FROM films
GROUP BY language

SELECT release_year, country, MAX(budget)
FROM films
GROUP BY release_year, country
ORDER BY release_year, country;



-- HAVING a great time, aLL together now

SELECT release_year,budget, gross
FROM films


SELECT release_year,budget, gross
FROM films
WHERE release_year>1990


SELECT release_year
FROM films
WHERE release_year>1990
GROUP BY release_year


SELECT release_year,
    AVG(budget) AS avg_budget,
    AVG(gross) AS avg_gross
FROM films
WHERE release_year>1990
GROUP BY release_year


SELECT release_year,
    AVG(budget) AS avg_budget,
    AVG(gross) AS avg_gross
FROM films
WHERE release_year>1990 
GROUP BY release_year
HAVING AVG(budget)>60000000


SELECT release_year,
    AVG(budget) AS avg_budget,
    AVG(gross) AS avg_gross
FROM films
WHERE release_year>1990 
GROUP BY release_year
HAVING AVG(budget)>60000000
ORDER BY AVG(gross) DESC



-- All together now (2)

-- select country, average budget, average gross
SELECT country, 
    AVG(budget) AS avg_budget, 
    AVG(gross) AS avg_gross
-- from the films table
FROM films
-- group by country 
GROUP BY country
-- where the country has a title count greater than 10
HAVING COUNT(films)>10
-- order by country
ORDER BY country
-- limit to only show 5 results
LIMIT 5



















