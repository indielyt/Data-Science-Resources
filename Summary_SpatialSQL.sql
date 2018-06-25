/*

UDEMY: SPATIAL SQL CLASS.


TABLE OF CONTENTS

--- Postgres, Postgis, and QGIS setup
	
	- Connecting postgres database to postgis
	- adding postgis layers to QGIS


--- A basic introduction to SQL with GIS Data

	-see all the data from one table
	-select all items meeting a conditional statement
	-create a new database table (qlayer) from an SQL query
	-delete a table (qlayer) from postgres database
	-Select parcels based on geometric intersection (with firm layer) and by firm zone (X)
	-Mulitple conditional statement selection


--- Sql data types, part 1: Numeric, Boolean, Text

	- Numerical operations

	- perform math operation, return as a value to temporary column (doesn't alter database)
	- math with multiple conditions
	- return single value (the average asmt value of the data)
	- Select with math in inner table (T1), with additional conditional statement
	- Perform statistical tests (compute coefficient of variation)

	- Boolean Operations

	- Character Operations

	- select by first letters of a field, then make new layer in database based on results
	- Concatenate strings
	- Convert data types
	- Multiple chained data type converions


--- Sql data types, part 2: Date and Time Operations, Spatial:

	--DATE AND TIME QUERIES:

	-Select date parts: The day attribute of the entire date, from the gps_date column in the trees table
	-Return the gps date and the name of the day (ie. Monday, Tuesday, etc) for trees table
	-Select based on date range
	-Perform math with dates: how long since...?
	-Complex queries based on dates and fields

	--SPATIAL QUERIES (INTRO): see docs for full list of spatial functions: https://postgis.net/docs/reference.html

	-Perform intersection of polygons and return new table: parks and parcels
	-Perform intersection of polygons and only return results
	-Perform intersection (spatial clip) of polygons with additional conditional statements: parcels and x500 flood zone


--- Traditional SQL: SELECT, WHERE, conditional expressions and GROUP BY 

	-SELECT
	-GROUP BY
	-MORE FUN WITH GROUP BY:
	-CASE STATEMENTS:
	-AGGREGATE FUNCTIONS:
	-OTHER SQL COMMANDS (ORDER, LIMIT, OFFSET, BETWEEN):
	-Changing Data: DROP, CREATE, INSERT, ALTER, UPDATE
	-Writing SQL Functions


-- Spatial SQL for Vector Geometry

	-Spatial Operations: coordinate system manipulation
	-Spatial operations: Adjacent, Buffer, Contains, Distance, Intersect, more..
	-Spatial operations: Topological overlay (ERASE, INTERSECT, IDENTITY)


-- Spatial SQL for Geographic Analysis

	-Spatial operations: Distance, Adjacency, and Interaction Matrices
	-Geographic Analysis: Nearest Neighbor Index


*/








/*
--- Postgres, Postgis, and QGIS setup

Connecting postgres database to postgis:
	asdf, see ch.2


postgres user/pw:
	user: postgres
	pw: -- my password for computer --

adding postgis layers to QGIS:
	-layer
	-add layer
	-add postgis layer
	-host: localhost
	-database: give the name of the database to connect to
	-username: 'postgres' for training exercise
	-pw: my password for this computer

*/










/* A basic introduction to SQL with GIS Data */


	--(In pgadmin4 gui which runs in browser)

	--see all the data from one table:
	SELECT * FROM parcels

	--select all items meeting a conditional statement
	SELECT propclass
	FROM parcels
	WHERE acres > 44

	--create a new database table (qlayer) from an SQL query
	SELECT *
	INTO qlayer
	FROM parcels
	WHERE acres>  44

	--delete a table (qlayer) from postgres database
	DROP TABLE qlayer

	--Select parcels based on geometric intersection (with firm layer) and by firm zone (X)
	SELECT parcels.* INTO qlayer
	FROM parcels, firm
	WHERE st_intersects(parcels.geometry, firm.geometry)
	AND firm.zone='X'

	--Mulitple conditional statement selection
	SELECT parcels.* INTO qlayer
	FROM parcels, firm, hydro
	WHERE st_intersects(parcels.geometry, firm.geometry)
	AND firm.zone = 'AE'
	AND parcels.asmt > 500000
	AND st_distance(parcels.geometry, hydro.geometry)  < 600












/* Sql data types, part 1: Numeric, Boolean, Text */

	--(In pgadmin, query tool)

	--Numberical Operations:

	--perform math operation, return as a value to temporary column (doesn't alter database)
	SELECT asmt - land AS structvalue, parcelkey
	FROM parcels

	--math with multiple conditions
	SELECT asmt / land AS myratio, parcelkey
	FROM parcels
	WHERE land > 0

	--return single value (the average asmt value of the data)
	SELECT avg(asmt)
	FROM parcels

	--Select with math in inner table (T1), with additional conditional statement
	SELECT * FROM
		(SELECT asmt / land AS VALUEPCTDIF, PARCELKEY
		FROM parcels
		WHERE land > 0) AS T1
	WHERE VALUEPCTDIF  > 1.25

	--Perform statistical tests (compute coefficient of variation)
	SELECT stddev(asmt) / avg(asmt) AS COEFFICIENTOFVARIATION
	FROM parcels




	--Boolean Operations:

	/*
	AND, OR, NOT, EQV, IMP, XOR
	greater than, less than, not equal to, equal to
	*/


	--Character Operations:

	--select by first letters of a field, then make new layer in database based on results
	SELECT * INTO qlayer
	FROM parcels WHERE LEFT(AddrStreet, 2) = 'BU'

	--Concatenate strings
	SELECT concat(addrno, ' ', AddrStreet) FROM parcels

	--Convert data types
	SELECT asmt::numeric FROM parcels

	--Multiple chained data type converions
	SELECT asmt::numeric::money::text FROM parcels











/* Sql data types, part 2: Date and Time Operations, Spatial */


	--DATE & TIME QUERIES:

	--Select date parts: The day attribute of the entire date, from the gps_date column in the trees table
	--See docs for complete formatting protocol: https://www.postgresql.org/docs/9.1/static/functions-formatting.html
	SELECT date_part('day', gps_date) FROM trees;
	SELECT date_part('month', gps_date) FROM trees;
	SELECT date_part('year', gps_date) FROM trees;

	--Return the gps date and the name of the day (ie. Monday, Tuesday, etc) for trees table
	SELECT gps_date, to_char(gps_date, 'Day') FROM trees

	--Select based on date range
	SELECT * FROM trees WHERE gps_date > '06/03/2001'

	--Perform math with dates: how long since...?
	SELECT inv_date - '06/03/2001' FROM trees

	--Complex queries based on dates and fields: Select maintenance and id fields where more than 400 days since inventory date
	SELECT site_id, maint
	FROM trees
	WHERE to_char(inv_date - '06/03/2001', 'DD')::numeric > 400


	--SPATIAL QUERIES (INTRO): see docs for full list of spatial functions: https://postgis.net/docs/reference.html


	--Perform intersection of polygons and return new table: parks and parcels
	DROP TABLE qlayer --removes the existing qlayer
	SELECT parcels.*
	INTO qlayer
	FROM parcels, parks
	WHERE st_intersects(parcels.geometry, parks.geometry)

	--Perform intersection of polygons and only return results
	SELECT parcels.parcelkey, parks.name
	FROM parcels, parks
	WHERE st_intersects(parcels.geometry, parks.geometry)

	--Perform intersection (spatial clip) of polygons with additional conditional statements: parcels and x500 flood zone
	--intersection is a spatial clip, intersects is a boolean true/false
	DROP TABLE qlayer;
	SELECT st_intersection(parcels.geometry, firm.geometry) AS geometry 
	INTO qlayer
	FROM parcels, firm
	WHERE firm.zone = 'X500'
	AND st_intersects(parcels.geometry, firm.geometry) --avoids computing intersection at all parcels (produces Null where no intersection)












/* Traditional SQL: SELECT, WHERE, conditional expressions and GROUP BY */











	--SELECT: 

	--general form 
	SELECT <something>
	FROM  <table>
	WHERE <condition>
	AND, OR, IN, >, <, =

	--general example
	SELECT parcels.*
	FROM parcels
	WHERE propclass = 210

		--SELECT does not change any data
		--(*) asterisk to select all columns in a table
	
	--joins through SELECT statements: 
	SELECT parcels.swis, propclas.description
	FROM parcels, propclass
	WHERE parcels.propclass = propclas.value::integer






	--GROUP BY:






	--simple
	SELECT sum(acres) AS sumacres, propclass --AS gives the new column name for the sum function
	FROM parcels
	GROUP BY propclass --GROUP BY defines on what category the aggegrate function is performed

	--multiple aggregate functions, returns sums(2), average(1), and propclass columns
	SELECT sum(acres) AS sumacres, 
		sum(asmt)::numeric::money AS sumasmt, --converts the sum to money
		avg(asmt)::numeric::money as avgasmt, 
		propclass 
	FROM parcels
	GROUP BY propclass

	--GROUP BY w/ COUNT, returns number of records grouped by propclass
	SELECT COUNT(*) AS numprops, propclass
	FROM parcels
	GROUP BY propclass

	--GROUP BY with spatial conditions
	SELECT sum(asmt) AS sumasmt, propclass
	FROM parcels, firm
	WHERE st_contains(firm.geometry, parcels.geometry) --where firm geometry contains parcels(?)
	AND firm.zone = 'X' --only in zone x firm zones
	GROUP BY propclass --sum based on propclass field
	ORDER BY propclass --return results in ascending order






	--MORE FUN WITH GROUP BY:






	--Setting up for merge(dissolve)...kind of confusing, but selecting based on 'left' function, which
	--takes the value in the identified columns, the 1st value from the left.  'right' is similar, only 
	--in this case selecting based on the 2 values from the right.  Result is that column is added of the 
	--hundreds (100, 200, 300, etc) which identifies which hundred group (base class) each propclass number is belongs.
	SELECT parcels.parcelkey, 
		parcels.propclas,
		propclas.description, 
		propclas.value
	FROM parcels, propclas
	WHERE left(parcels.propclass::text, 1) = left(propclas.value, 1)
	AND right(propclas.value, 2) = '00'
	ORDER BY value


	--Union of geometries based on base class (see above)
	DROP TABLE qlayer;
	SELECT st_union(geometry) AS geometry, propclas.description --st_union is the aggregation function for group by command
	INTO qlayer
	FROM parcels, propclas
	WHERE left(parcels.propclass::text, 1) = left(propclas.value, 1)
	AND right(propclas.value, 2) = '00'
	GROUP BY propclas.description





	--CASE STATEMENTS:





	-- Perform mathematical operations to return different values (effectivly an if/then statement)
	SELECT parcelkey, asmt,
		CASE
			WHEN asmt = 0 THEN 0
			WHEN asmt BETWEEN 1 AND 100000 THEN asmt * 0.07
			WHEN asmt BETWEEN 100001 AND 500000 THEN asmt * 0.09
			ELSE asmt * 0.11
		END AS taxbill
	FROM parcels



	-- Case statements with geographic operations
	DROP TABLE qlayer;                                       -- delete qlayer before recreating it in the case statement
	SELECT zone,											 -- select from the zone column (from firm table - see below)
		CASE 												 -- begin case statement
			WHEN zone = 'X' THEN st_buffer(geometry, 100)    -- buffer the X zone with 100 feet
			WHEN zone = 'AE' THEN st_buffer(geometry, 200)	 -- buffer the AE zone with 200 feet
			WHEN zone = 'X500' THEN st_buffer(geometry, 300) -- buffer the X500 zone iwth 300 feet
			ELSE geometry									 -- provide else statement for other zone delineations
		END AS geometry										 -- end case statement
	INTO qlayer                                              -- create new layer from buffers
	FROM firm                                                -- end of select statement, from firm table






	-- AGGREGATE FUNCTIONS: https://www.postgresql.org/docs/9.5/static/functions-aggregate.html





	-- Basic Aggregation: Average
	SELECT avg(Ob_1995) AS avg_obesity_1995 FROM states

	-- Statistical funcations:  Pearson's correlation coefficient
	SELECT corr(ob_1995, ob_2000) AS pearson FROM states


	-- Spatial Aggregation: Mean center of the united state's states
	DROP TABLE qlayer;
	SELECT st_point(avg(ST_Centroid(geometry))),      --st_point creates pont file, ST_Centroid finds center of each state's geometry
		avg(ST_Y(ST_Centroid(geometry))) AS geometry  --avg finds average of all state's x/y centroid coordinates
	INTO qlayer
	FROM states






	-- OTHER SQL COMMANDS (ORDER, LIMIT, OFFSET, BETWEEN):





	
	-- ORDER BY:  returning the results in ASCENDING list
	SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009

	-- ORDER BY:  returning the results in DESCENDING list
	SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009 DESC

	-- LIMIT: return a set number of results
	SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009
	LIMIT 10

	--OFFSET: skip a determined number of results before returning query
	SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009
	OFFSET 20
	LIMIT 10

	--IN: return results in a list or another query
	SELECT ob_2009, name
	FROM states
	WHERE name IN(SELECT name FROM states WHERE left(name,1) = 'M') -- return results that start with the letter M

	-- BETWEEN: return values between a two numbers
	SELECT ob_2009, name
	FROM states
	WHERE ob_2009 BETWEEN 25 AND 30

	--UNION ALL:  Concatenate multipe queries together in a return result
	(SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009 DESC
	LIMIT 10)

	Union all

	(SELECT ob_2009, name
	FROM states
	ORDER BY ob_2009 DESC
	LIMIT 10)

	--GROUP BY, TEMPORARY TABLES, LIMIT, all used together
	SELECT name FROM 										-- outer select start, querying only the name
		(SELECT name, count(name) AS numstates FROM 	-- inner(1) select start
			(											-- inner(2) select start
				(SELECT ob_2009, name					
				FROM states
				ORDER BY ob_2009 DESC
				LIMIT 10)

				Union all

				(SELECT ob_2009, name
				FROM states
				ORDER BY ob_2009 DESC
				LIMIT 10))								-- inner(2) select end
				AS T1									-- Define inner(2) result as TI (temporary table in memory)
		GROUP BY name									-- Group results by their names (note multiple years queried, same states potentially show up multiple times)
		) 												-- inner(1) select end
		AS T2											-- Define inner(1) result as T2
	WHERE numstates=2									-- outer select end, only returning results where numstates=3
	-- returns the name of states that show up twice in the inner(2) query


	-- Return the above geometries
	--GROUP BY, TEMPORARY TABLES, LIMIT, all used together
	DROP TABLE qlayer;									-- Drop existing table

	SELECT name, geometry 								-- Begin Select in to geospatial layer(qlayer)
	INTO qlayer
	FROM states WHERE name IN 

	(SELECT name FROM 										-- outer select start, querying only the name
		(SELECT name, count(name) AS numstates FROM 	-- inner(1) select start
			(											-- inner(2) select start
				(SELECT ob_2009, name					
				FROM states
				ORDER BY ob_2009 DESC
				LIMIT 10)

				Union all

				(SELECT ob_2009, name
				FROM states
				ORDER BY ob_2009 DESC
				LIMIT 10))								-- inner(2) select end
				AS T1									-- Define inner(2) result as TI (temporary table in memory)
		GROUP BY name									-- Group results by their names (note multiple years queried, same states potentially show up multiple times)
		) 												-- inner(1) select end
		AS T2											-- Define inner(1) result as T2
	WHERE numstates=2									-- outer select end, only returning results where numstates=3
	)






	--Changing Data: DROP, CREATE, INSERT, ALTER, UPDATE





	--CREATE/DROP: create and delete table from database
	--First create a table named 'mytable' with a text column called 'name' and 
	--a geometry table called 'geometry' in the #2261 coordinate system (http://spatialreference.org/).
	CREATE TABLE mytable(name text, geometry geometry(Geometry, 2261));
	DROP TABLE mytable;



	--INSERT INTO: put new data into table
	INSERT INTO mytable (name, geometry)                  --insert into name and geometry column of my table the following line
	SELECT name, geometry FROM parks WHERE size > 1;      --name and geometries of park table where size > 1



	--ALTER TABLE:  add a column
	ALTER TABLE mytable
	ADD Column parksize double precision;



	--UPDATE TABLE: change values based on common name between tables
	UPDATE mytable						--update mytable
	SET parksize = parks.size 			--set the parksize column of mytable to the size column of parks table
	FROM parks
	WHERE parks.name = mytable.name;    --only where shared names exist






	--Writing SQL Functions




	--Create an SQL function
	CREATE FUNCTION getfloodgeom(x text)    --Define the function name (getfloodgeom) and input (text variable)
	RETURNS TABLE (mygeom geometry)   		--Define table of results (mygeom column with geometry of query)
	$$
		SELECT st_intersection(parcels.geometry, firm.geometry) AS geometry   							--Select the intersection of parcels and firm
 		FROM parcels, firm WHERE st_intersects(parcels.geometry, firm.geometry) AND firm.zone = $1;  	--where parcels, firm intersect and firm zone = first input variable ($1 = x text)
	$$ LANGUAGE SQL;

	SELECT getfloodgeom('AE') INTO qlayer	--Call getfloodgeom function and put results into qlayer.

















/* Spatial SQL for Vector Geometry */
-- Importing shapefiles into postgres (https://stackoverflow.com/questions/40636158/import-shape-file-into-postgis-with-pgadmin-4)















	--Spatial Operations: coordinate system manipulation
	--SRID = Spatial Reference Identification


	--Find projection of a layer (www.spatialreference.org)
	SELECT ST_SRID(geometry) FROM states2


	--Define projection of a spatial table that is missing (or has incorrect) spatial coordinate system
	SELECT UpdateGeometrySRID('states2', 'geometry', 2796)


	--Transform (project) to new coordinate system (only works on geometry column!!!! Doesn't alter SRID number)
	SELECT ST_Transform(geometry, 3450) FROM states2

	--Transform (project) into a new layer
	SELECT name, ST_Transform(geometry, 3450) AS geometry INTO states3
	FROM states2;

	--Project to new coordinate system (changes existing layer)
	ALTER TABLE states2
		ALTER COLUMN geometry
		TYPE Geometry(Multipolygon, 2959)
		USING ST_TRANSFORM(geometry, 2959);





	--Spatial operations: Adjacent, Buffer, Contains, Distance, Intersect, more..
	




	--ADJACENCY: Checking what geometries touch one another.  Returns qlayer with all parcels adjacent to the selected parcelkey
	DROP TABLE qlayer;
	SELECT * INTO qlayer FROM parcels
	WHERE st_touches(parcels.geometry,
		(SELECT geometry
		FROM parcels
		WHERE parcelkey='50070006200000010150000000'))


	--ADJACENCY WITH MATHS: Find landvalues (asmt-land) of adjacent parcels
	SELECT sum(asmt)::numeric::money - sum(land)::numeric::money 
	FROM parcels
	WHERE st_touches(parcels.geometry,
		(SELECT geometry
		FROM parcels
		WHERE parcelkey='50070006200000010150000000'))


	--ADJACENCY WITH RETURN FORMATTING:
	SELECT addrno || ' ' || addrstreet AS address
	FROM parcels
	WHERE st_touches(parcels.geometry,
		(SELECT geometry
		FROM parcels
		WHERE parcelkey='50070006200000010150000000'))


	--BUFFER:
	SELECT parcels.parcel_id, st_buffer(geometry, 100) AS geometry
	INTO qlayer
	FROM parcels
	WHERE parcelkey='50070006200000010150000000'


	--CONTAINS (where parcels geometry if fully contained by firm 'AE' geometry):
	DROP TABLE qlayer;
	SELECT parcels.*
	INTO qlayer
	FROM parcels, firm
	WHERE st_contains(firm.geometry, parcels.geometry)
	AND firm.zone = 'AE'


	--INTERSECTS (where parcels and firm 'AE' geometry intersect (result include partial coverage of parcel by firm):
	DROP TABLE qlayer;
	SELECT parcels.*
	INTO qlayer
	FROM parcels, firm
	WHERE st_intersects(firm.geometry, parcels.geometry)
	AND firm.zone = 'AE'


	--DISTANCE (between two geometries):
	SELECT st_distance(parcels.geometry, firm.geometry)::integer as dist, parcels.parcel_id
	FROM parcels, firm
	WHERE st_distance(parcels.geometry, firm.geometry) < 300
	AND zone = 'X'


	--DISTANCE (using DWithin, which uses indexes and is significantly faster than st_distance)
	SELECT st_distance(parcels.geometry, firm.geometry)::integer as dist, parcels.parcel_id
	FROM parcels, firm
	WHERE st_DWithin(parcels.geometry, firm.geometry, 300)
	AND zone = 'X'





	--Spatial operations: Topological overlay (ERASE, INTERSECT, IDENTITY)




	--ERASE (erase the middle geometry from the leftsquare)
	DROP TABLE qlayer;
	SELECT leftsquare.side, st_difference(leftsquare.geometry, middle.geometry) AS geometry
	INTO qlayer
	FROM leftquare, middle


	--INTERSECT
	DROP TABLE qlayer;
	SELECT leftsquare.side AS l_side, rightsquare.side AS r_side, --imbues attributes (side) into qlayer as l_side and r_side columns
		st_intersection(leftsquare.geometry rightsquare.geometry) as geometry
	INTO qlayer
	FROM leftsquare, rightsquare
	WHERE st_intersects(leftsquare.geometry, rightsquare.geometry) --speeds up query, prevent returning null values


	--IDENTITY (takes all of first feature, and attributes of overlapping 2nd feature geometry)(re-uses a lot of the above)
	--See Esri documentation for explanation of identity
	DROP TABLE qlayer;
	SELECT * INTO qlayer
	FROM
		--Get the intersecting geometries and attributes
		(SELECT leftsquare.side AS l_side, rightsquare.side AS r_side, 
		st_intersection(leftsquare.geometry rightsquare.geometry) as geometry
		FROM leftsquare, rightsquare
		WHERE st_intersects(leftsquare.geometry, rightsquare.geometry) --speeds up query, prevent returning null values
	
		UNION ALL --union of 

		--Get the non intersection geometries of 1st feature (leftsquare)
		SELECT leftsquare.side AS l_side, ' ' AS r_side, --no return from r_side (2nd feature)
			st_intersection(leftsquare.geometry rightsquare.geometry) as geometry
		FROM leftsquare, rightsquare
		WHERE st_intersects(leftsquare.geometry, rightsquare.geometry) 
		) AS T1 --Virtual table




















/* Spatial SQL for Geographic Analysis  */




















	--Spatial operations: Distance, Adjacency, and Interaction Matrices




	--Distance as a table
	SELECT a.name, b.name, (st_distance(a.geometry, b.geometry, true) * 0.00062)::text AS dist
	FROM upstate AS a, upstate AS b
	--returns a table of distances between all cities, however multiple repeated values
	--SELECT statement returns three columns:  name(a), name(b), distance(true calculates distance along elipse)
	--FROM statement identifies the sourc of (a) and (b) in SELECT statement


	--Distance as a symmetrical  matrix of distances between cities
	--Crosstab (https://www.postgresql.org/docs/9.2/static/tablefunc.html) - functions as a pivot table
	--ct_row must be manually defined by SELECT name FROM upstate ORDER BY name
	SELECT * FROM
	Crosstab('SELECT a.name::text, b.name::text,
				(st_distance(a.geometry, b.geometry, true) * 0.00062)::text
			FROM upstate AS a, upstate AS b
			ORDER BY 1, 2'
			) AS 
			ct_row(row_name text, Auburn text, Bingamton text, Elmira text,
				Ithaca text, Rochester text, Syracuse text)


	--Adjacency: Returning symmetrical matrix of boolean values indicating adjacency (defined by us as <50mi apart)
	SELECT * FROM
	Crosstab('SELECT a.name::text, b.name::text,
				CASE
					WHEN st_distance(a.geometry, b.geometry, true)*0.00062 < 50 THEN 1::text
					ELSE 0::text
					END AS dist
				FROM upstate AS a, upstate AS b
				ORDER BY 1, 2'
			) AS 
			ct_row(row_name text, Auburn text, Bingamton text, Elmira text,
				Ithaca text, Rochester text, Syracuse text)


	--Adjacency: Touching
	--Note: couldn't get this one to work, despite copying code from video
	SELECT * FROM
	Crosstab('SELECT a.name::text, b.name::text,
			st_touches(a.geometry, b.geometry)::text AS tt
			FROM states AS a, states AS b
			WHERE a.name in ("Alabama", "California", "Nevada", "Oregon", "Mississippi")
			AND   b.name in ("Alabama", "California", "Nevada", "Oregon", "Mississippi")
			ORDER BY 1,2'
			) AS
			ct(row_name text, Alabama text, California text, Mississippi text,
				Nevada text, Oregon text);



	--Interaction via inverse distance weighting
	SELECT * FROM
	Crosstab('SELECT a.name::text, b.name::text, CASE
				WHEN st_distance(a.geometry, b.geometry, true)*0.00062 = 0 THEN 0::text
				ELSE (1/(st_distance(a.geometry, b.geometry, true)*0.00062))::text 
				END AS dist
			FROM upstate AS a, upstate AS b
			ORDER BY 1, 2'
			) AS 
			ct(row_name text, Auburn text, Bingamton text, Elmira text,
				Ithaca text, Rochester text, Syracuse text)







	--Geographic Analysis: Nearest Neighbor Index 





	--Return table of distances between cities, ordered by aname, then distances between cities
	SELECT st_distance(a.geometry, b.geometry, true) * 0.00062 AS dist,
		a.name AS aname, b.name AS bname
	FROM upstate AS a, upstate AS b
	WHERE a.name <> b.name
	ORDER BY aname, dist


	--Return only the distance to the closest city
	SELECT aname, min(dist)
	FROM 
	(SELECT st_distance(a.geometry, b.geometry, true) * 0.00062 AS dist,
		a.name AS aname, b.name AS bname
	FROM upstate AS a, upstate AS b
	WHERE a.name <> b.name
	ORDER BY aname, dist ASC) AS T1
	GROUP BY aname






















