/*

UDEMY: SPATIAL SQL CLASS.


TABLE OF CONTENTS

-- GETTING THINGS SETUP 

	-Create New Database
	-Establish postgis Connection
	-Importing .shp and .dbf with PostGIS Utilities


-- MANAGING DATA AND USERS

	-Create Roles
	-Granting Privileges
	-Assigning Logins
	-Adding Constraints to Protect Data
	-Creating Views to Visualize our Data
	-Loading Raster Data
	-Editing Data in QGIS (no detail provided below)
	-Multi-User Editing (no detail provided below)
	-Creating Triggers

-- DEPLOYING THE ENTERPRISE GIS

	-Server side analysis of PostGRES data
	-Creating external programs to access PostGRES
	-A basic introduction to internet map servers with QGIS plugin



*/














/*   GETTING THINGS SETUP


-- (I started with Postgres 10 and pgadmin 4 already installed). 
	--Create new database (right clicking the postgresql 10 server)
	--Establish connection to postgis functions by running the following sql query:
		CREATE EXTENSION postgis;

-- Importing Shapefile in to Postgres
	--Use prj2epsg.org/search to convert .prj file to srid (spatial reference id)
		--NOte: postgres does not automatically detect spatial reference of shapefiles
	-- Use the postgis utility, pgShapeLoader.app, to load in the data (This was in the 
	  	applications folder, POSTgis Utilities folder)
	-- With the srid code, manually change that column in pgShapeLoader window to the correct number
	--In QGIS, Add PostGIS Layer, connect to the database (with a chosen connection name, host (localhost), 
	  database name, and username/password).  Choose the layers (select radio button for
	  show layers with no spatial geometry).

*/















/* MANAGING DATA AND USERS

--Create Roles
	--In each server, there is a Roles item, which can be right clicked to create new role.  
	  we created a number of roles which will eventually have different roles. Note that once
	  we give the role a name in the create role window, we can scroll to the right and click
	  sql to see the sql command to create new role.  Using the following sql query, we've done
	  the same task in sql: */
	CREATE ROLE sdf WITH
	NOLOGIN
	NOSUPERUSER
	NOCREATEDB
	NOCREATEROLE
	INHERIT
	NOREPLICATION
	CONNECTION LIMIT -1; /*

-- Granting Privileges
	--Under the database, schema, public, right click tables and chose grant wizard. Use the drop downs
	  and window guidance to select tables, groups, and privileges for each group.  NOte that 
	  'with grant option' will allow that user to additionally grant privileges to other users.







-Assigning Logins
	--Using the same create new role from above, we can create roles for individual users.  Assign a name and
	  password if needed (we use the name as a password for the exercies), then assign privleges via membership
	  to the specific groups we created earlier.

	--Back in QGIS, we can add a PostGIS layer, but instead of adding as postgres user, create new connection
	  with any one of the new logins we just created (and the associated name and password).  
	  Mo will be able to do anything, but Larry can't delete 
	  anything.  Same with the other viewer logins.  Other users (and their connections) will be similarly limited
	  or empowered to see/edit/move/save/delete data as allowed in the group privileges.







-Adding Constraints to Protect Data
	--From a table in the database, right click, add new object, add new check.  In the window that pops up,
	  add an SQL query that defines the limitations as follows (note: uses an sql queries in the course resources,
	  queries.txt) 
	--Correction... PGAdmin4 doesn't have the add check option.  The sql query below can be run with the simple query
	  tool */
	  ALTER TABLE parcels2007
	  	ADD CONSTRAINT school
	  	CHECK ("SCHDIST" IN (	'',
		'CANDOR CENTRAL SCHOOL DISTRICT',
		'CORTLAND CENTRAL SCHOOL DISTRICT',
		'DRYDEN CENTRAL SCHOOL DISTRICT',
		'GEORGE JR CENTRAL SCHOOL DISTRICT',
		'GROTON CENTRAL SCHOOL DISTRICT',
		'HOMER CENTRAL SCHOOL DISTRICT',
		'ITHACA CITY SCHOOL DISTRICT',
		'LANSING CENTRAL SCHOOL DISTRICT',
		'MORAVIA CENTRAL SCHOOL DISTRICT',
		'NEWARK VALLEY CENTRAL SCHOOL DISTRICT',
		'NEWFIELD CENTRAL SCHOOL DISTRICT',
		'ODESSA-MONTOUR CENTRAL SCHOOL DISTRICT',
		'SOUTHERN CAYUGA CENTRAL SCHOOL DISTRICT',
		'SPENCER-VAN ETTEN CENTRAL SCHOOL DISTRICT',
		'TRUMANSBURG CENTRAL SCHOOL DISTRICT'))
	--Attempts to edit the data will result in an error (won't be able to save the changes in pgadmin, nor a GIS editor) /*






/*
--Creating Views to Visualize our Data
	--Views are essentially virtual layers that are created from existing tables, or joins of multiple 
	  tables
	--In pgadmin, under schema, public, right click views and create new. 
		-give a name: resparcels (residential parcels)
		-For definition, use a sql query to select only the residential parcels (210) from the parcels2007 layer */
		SELECT * FROM parcels2007 
        WHERE pc='210'  
        /* 
        -Provide an owner, and under security assign priveleges to users and specific privileges (ie. read/write/select, etc)
        -Once the view is created, it can be imported into QGIS, or queried in pgadmin

    --Joins can also be used as views
    	-example, joining description of flood zones to the flood zone geometry through a view (same process as above) */
    	SELECT floodzone.geom, floodzone.zone, flood.desc
		FROM floodzone, flood
		WHERE floodzone.zone = flood.zone

	--Spatial Joins can also be used as views.  Here st.contains postgis function selects parcels in the fall creek watershed:
		SELECT parcels2007.*
		FROM parcels2007, watersheds
		WHERE st_contains(watersheds.geom,parcels2007.geom)
		AND watersheds.watershed = 'Fall Creek'







/*
--Loading Raster Data
	--No plugin is available for pgadmin 3 or 4 to import raster, but we can run the raster2pgsql from the bin folder. 
	--In terminal (or command line windows), navigate to the bin folder.  In my mac, it was here:
		/Library/PostgreSQL/10/bin
		Command Line Reminder: list directories (ls), change directories (cd), launch .exec (./raster2pgsql)
	--Run the raster2pgsql by entering : ./raster2pgsql
		--a list of options for the package will appear
				Daniels-MBP:bin Daniel$ ./raster2pgsql
				RELEASE: 2.4.4 GDAL_VERSION=20 (r16526)
				USAGE: raster2pgsql [<options>] <raster>[ <raster>[ ...]] [[<schema>.]<table>]
				  Multiple rasters can also be specified using wildcards (*,?).

				OPTIONS:
				  -s <srid> Set the SRID field. Defaults to 0. If SRID not
				     provided or is 0, raster's metadata will be checked to
				     determine an appropriate SRID.
				  -b <band> Index (1-based) of band to extract from raster. For more
				      than one band index, separate with comma (,). Ranges can be
				      defined by separating with dash (-). If unspecified, all bands
				      of raster will be extracted.
				  -t <tile size> Cut raster into tiles to be inserted one per
				      table row. <tile size> is expressed as WIDTHxHEIGHT.
				      <tile size> can also be "auto" to allow the loader to compute
				      an appropriate tile size using the first raster and applied to
				      all rasters.
				  -P Pad right-most and bottom-most tiles to guarantee that all tiles
				     have the same width and height.
				  -R  Register the raster as an out-of-db (filesystem) raster. Provided
				      raster should have absolute path to the file
				 (-d|a|c|p) These are mutually exclusive options:
				     -d  Drops the table, then recreates it and populates
				         it with current raster data.
				     -a  Appends raster into current table, must be
				         exactly the same table schema.
				     -c  Creates a new table and populates it, this is the
				         default if you do not specify any options.
				     -p  Prepare mode, only creates the table.
				  -f <column> Specify the name of the raster column
				  -F  Add a column with the filename of the raster.
				  -n <column> Specify the name of the filename column. Implies -F.
				  -l <overview factor> Create overview of the raster. For more than
				      one factor, separate with comma(,). Overview table name follows
				      the pattern o_<overview factor>_<table>. Created overview is
				      stored in the database and is not affected by -R.
				  -q  Wrap PostgreSQL identifiers in quotes.
				  -I  Create a GIST spatial index on the raster column. The ANALYZE
				      command will automatically be issued for the created index.
				  -M  Run VACUUM ANALYZE on the table of the raster column. Most
				      useful when appending raster to existing table with -a.
				  -C  Set the standard set of constraints on the raster
				      column after the rasters are loaded. Some constraints may fail
				      if one or more rasters violate the constraint.
				  -x  Disable setting the max extent constraint. Only applied if
				      -C flag is also used.
				  -r  Set the constraints (spatially unique and coverage tile) for
				      regular blocking. Only applied if -C flag is also used.
				  -T <tablespace> Specify the tablespace for the new table.
				      Note that indices (including the primary key) will still use
				      the default tablespace unless the -X flag is also used.
				  -X <tablespace> Specify the tablespace for the table's new index.
				      This applies to the primary key and the spatial index if
				      the -I flag is used.
				  -N <nodata> NODATA value to use on bands without a NODATA value.
				  -k  Skip NODATA value checks for each raster band.
				  -E <endian> Control endianness of generated binary output of
				      raster. Use 0 for XDR and 1 for NDR (default). Only NDR
				      is supported at this time.
				  -V <version> Specify version of output WKB format. Default
				      is 0. Only 0 is supported at this time.
				  -e  Execute each statement individually, do not use a transaction.
				  -Y  Use COPY statements instead of INSERT statements.
				  -G  Print the supported GDAL raster formats.
				  -?  Display this help screen.
	--Run ./raster2pgsql -G
		--a list of raster formats that can be imported will appear
				Daniels-MBP:bin Daniel$ ./raster2pgsql -G
				Supported GDAL raster formats:
				  Virtual Raster
				  GeoTIFF
				  National Imagery Transmission Format
				  Raster Product Format TOC format
				  ECRG TOC format
				  Erdas Imagine Images (.img)
				  CEOS SAR Image
				  CEOS Image
				  JAXA PALSAR Product Reader (Level 1.1/1.5)
				  Ground-based SAR Applications Testbed File Format (.gff)
				  ELAS
				  Arc/Info Binary Grid
				  Arc/Info ASCII Grid
				  GRASS ASCII Grid
				  SDTS Raster
				  DTED Elevation Raster
				  Portable Network Graphics
				  JPEG JFIF
				  In Memory Raster
				  Japanese DEM (.mem)
				  Graphics Interchange Format (.gif)
				  Graphics Interchange Format (.gif)
				  Envisat Image Format
				  Maptech BSB Nautical Charts
				  X11 PixMap Format
				  MS Windows Device Independent Bitmap
				  SPOT DIMAP
				  AirSAR Polarimetric Image
				  RadarSat 2 XML Product
				  PCIDSK Database File
				  PCRaster Raster File
				  ILWIS Raster Map
				  SGI Image File Format 1.0
				  SRTMHGT File Format
				  Leveller heightfield
				  Terragen heightfield
				  USGS Astrogeology ISIS cube (Version 3)
				  USGS Astrogeology ISIS cube (Version 2)
				  NASA Planetary Data System
				  MIPL VICAR file
				  EarthWatch .TIL
				  ERMapper .ers Labelled
				  NOAA Polar Orbiter Level 1b Data Set
				  FIT Image
				  GRIdded Binary (.grb)
				  Raster Matrix Format
				  OGC Web Coverage Service
				  OGC Web Map Service
				  EUMETSAT Archive native (.nat)
				  Idrisi Raster A.1
				  Intergraph Raster
				  Golden Software ASCII Grid (.grd)
				  Golden Software Binary Grid (.grd)
				  Golden Software 7 Binary Grid (.grd)
				  COSAR Annotated Binary Matrix (TerraSAR-X)
				  TerraSAR-X Product
				  DRDC COASP SAR Processor Raster
				  R Object Data Store
				  OziExplorer .MAP
				  Portable Pixmap Format (netpbm)
				  USGS DOQ (Old Style)
				  USGS DOQ (New Style)
				  ENVI .hdr Labelled
				  ESRI .hdr Labelled
				  Generic Binary (.hdr Labelled)
				  PCI .aux Labelled
				  Vexcel MFF Raster
				  Vexcel MFF2 (HKV) Raster
				  Fuji BAS Scanner Image
				  GSC Geogrid
				  EOSAT FAST Format
				  VTP .bt (Binary Terrain) 1.3 Format
				  Erdas .LAN/.GIS
				  Convair PolGASP
				  Image Data and Analysis
				  NLAPS Data Format
				  Erdas Imagine Raw
				  DIPEx
				  FARSITE v.4 Landscape File (.lcp)
				  NOAA Vertical Datum .GTX
				  NADCON .los/.las Datum Grid Shift
				  NTv2 Datum Grid Shift
				  CTable2 Datum Grid Shift
				  ACE2
				  Snow Data Assimilation System
				  KOLOR Raw
				  ROI_PAC raster
				  Azavea Raster Grid format
				  Swedish Grid RIK (.rik)
				  USGS Optional ASCII DEM (and CDED)
				  GeoSoft Grid Exchange Format
				  Northwood Numeric Grid Format .grd/.tab
				  Northwood Classified Grid Format .grc/.tab
				  ARC Digitized Raster Graphics
				  Standard Raster Product (ASRP/USRP)
				  Magellan topo (.blx)
				  Rasterlite
				  SAGA GIS Binary Grid (.sdat)
				  Kml Super Overlay
				  ASCII Gridded XYZ
				  HF2/HFZ heightfield raster
				  Geospatial PDF
				  OziExplorer Image File
				  USGS LULC Composite Theme Grid
				  Arc/Info Export E00 GRID
				  ZMap Plus Grid
				  NOAA NGS Geoid Height Grids
				  MBTiles
				  IRIS data (.PPI, .CAPPi etc)
				  Planet Labs Mosaics API
				  GeoPackage
				  Planet Labs Scenes API
				  HTTP Fetching Wrapper
	--Importing the Raster data: (we'll run the following from the same bin folder)
		./raster2pgsql -s 26918 /Users/Daniel/Documents/Programming/SQL/UDEMY_OpenSourceEnterpriseGIS/Supportmaterial/layers/*.dem public.mydem >/Users/Daniel/Downloads/out.sql
		--This follows the 'USAGE' directions above
		--Produces an sql query to help us import the raster (we haven't imported the raster yet)
		--Now we have to launch postgres from the same command line and directory, and run the out.sql file (which give instructions on importing the raster):
			psql -U postgres -d OpenSourceEnterpriseGIS -f /Users/Daniel/Downloads/out.sql
			- psql opens postgres, -U is the user name prompt , -d is the database name prompt, -f is the file we just created above

			-hit a roadblock, trying to allow authentication is terminal with help from here: https://stackoverflow.com/questions/2942485/psql-fatal-ident-authentication-failed-for-user-postgres
			-Not sure how exactly I fixed this, but was able to login by this method: adding login name and password from command line: */
				psql -d OpenSourceEnterpriseGIS  -U postgres  /*
			-Also changes the pg_hba.conf permissions from md5 to trust (should change back for better security)
			-Also added the the postgres bin file to the PATH variable by methods in this post: https://stackoverflow.com/questions/36155219/psql-command-not-found-mac
			-Was able to run the raster import command from above and it worked.
			-This post helped with psql command line: https://alvinalexander.com/blog/post/postgresql/log-in-postgresql-database

			-Note: making hidden files viewable: defaults write com.apple.finder AppleShowAllFiles YES  ... or NO if you want to hide the hidden files
			  then relaunch file explorer

	--Viewing Raster in QGIS:  
		-First, connect to database (either through add layer, add postgis layer, or POSSIBLY through the DB Manager)
		-In DB Manager (under databases tab), drill down to the dem table in public schema
		-Find raster, right click, add to canvas







--Creating Triggers
	-A trigger is a specification that the database should automatically execute a particular
	 function whenever a certain type of operation is performed and can be attached to both
	 tables and views.
	-Triggers 'fire' when certain events occur.  We'll create a function that runs when a 
	 certain event occurs
	-Triggers can be created through code: */

--Create a function in our database, in the selected schema, with a name upate_parcels	
CREATE OR REPLACE FUNCTION update_parcels()
RETURNS trigger AS $update_parcels$
	BEGIN
		--Check for the property class, must have value otherwise raise exception
		IF NEW.propclass IS NULL THEN
			RAISE EXCEPTION 'error: propclass field cannot be empty';
		END IF;

		--Retrieve the name of the watershed automatically, NOTE: watershed is a column in watersheds table
		--ST.Within is a postgis function. LIMIT 1 returns only 1 record
		NEW.watershed = (SELECT watershed as watershed FROM watersheds
		WHERE ST.Within(NEW.geom, watersheds.geom) LIMIT 1);

		--If the above retrieve statment returned a null value, then we know the new
		--parcel is not within a watershed defined in our data, which is required for creating new parcels
		IF NEW.watershed IS NULL THEN
			RAISE EXCEPTION 'error: parcels must be within a watershed';
		END IF;

		--Prepare to add the record
		RETURN NEW;
	END;
$update_parcels$ LANGUAGE PLpgsql;

--If the trigger update_parcels already exists on parcels2007, we'll drop the existing trigger and 
--create a new trigger on the parcels2007 layer.  We'll execute the trigger for every row (new
--parcel) that is created to ensure we get parcels with propclass values and parcels within a watershed
DROP TRIGGER IF EXISTS update_parcels ON parcels2007;
CREATE TRIGGER update_parcels BEFORE INSERT OR UPDATE ON parcels2007
	FOR EACH ROW EXECUTE PROCEDURE update_parcels();
--NOTE: might have to run this SQL query twice in postgis when the trigger doesn't already exist...due to the 
--drop trigger line.
















/* DEPLOYING THE ENTERPRISE GIS

-Server side analysis of PostGRES data
	-Through DB Manager in QGIS, we can run SQL queries directly in the database, load them as virtual 
	-layers in QGIS.  See the queries.txt file for example sql queries using postgis functions
	-Allows faster queries (being backend side rather that gis client side)
	







-Creating external programs to access PostGRES
	-The tutorial uses the psycopg python package to interact with PostGRES programatically
		-http://initd.org/psycopg/download/ OR conda install psycopg2 in the Anaconda Environment
	-A python script opens a database connection, prompts the user for a parcelkey, runs the postgis 
		query in the background and returns a result.  (Note, user input not supported in sublime, but 
		this worked flawlessly running the tabletop.py script from the command line).  the script is as follows:

			import psycopg2

			#DB connection properties
			conn = psycopg2.connect(dbname = 'OpenSourceEnterpriseGIS', host= 'localhost', port= 5432, user = 'postgres',password= 'one11D9*')
			conn.autocommit = True
			cur = conn.cursor()  ## open a cursor


			# For production, the user will input the parcelkey
			if True:
				acctid = repr(str(input("Enter your Account ID: ")))

				thesql = "SELECT min(st_distance(floodzone.geom,parcels2007.geom)) as dist " \
				         "FROM floodzone, parcels2007 WHERE parcels2007.parcelkey = " + acctid + " AND " \
				         "floodzone.zone = 'AE' "


			# For testing, have parcel ID hard coded into parcelkey sql
			if False:
				thesql = "SELECT min(st_distance(floodzone.geom,parcels2007.geom)) as dist " \
				         "FROM floodzone, parcels2007 WHERE parcels2007.parcelkey = '50308907200000010011090000' AND " \
				         "floodzone.zone = 'AE' "

			#print thesql
			cur.execute(thesql)
			rows = cur.fetchall()
			for row in rows:
			    print ("Distance to the closest AE flood zone is:   ", row[0])
			    
			cur.close()

			acctid = repr(str(input("Hit return to end")))







-A basic introduction to internet map servers with QGIS plugin
	-









