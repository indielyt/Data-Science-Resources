# Datacamp INTERACTIVE DATA VISUALIZATION WITH BOKEH

# Chapter 1: Basic plotting with Bokeh
	
		# A simple scatter plot
		# A scatter plot with different shapes
		# Customizing your scatter plots
		# Lines
		# Lines and markers
		# Patches
		# Plotting data from NumPy arrays
		# Plotting data from Pandas DataFrames
		# The Bokeh ColumnDataSource (continued)
		# Selection and non-selection glyphs
		# Hover glyphs
		# ColorMapping


# Chapter 2: Layouts, Interactions, and Annotations

		# Creating Rows of Plots
		# Creating columns of plots
		# Nesting rows and columns of plots
		# Creating gridded layouts
		# Starting tabbed layouts
		# Displaying tabbed layouts
		# Linked axes
		# Linked brushing
		# How to create legends
		# Positioning and styling legends (location & background color)
		# Adding a hover tooltip


# Chapter 3: Building interactive apps with Bokeh

		# Using the current document
		# Add a single slider
		# Multiple sliders in one document
		# How to combine Bokeh models into layouts
		# Learn about widget callbacks
		# Updating data sources from dropdown callbacks (ALL CODE LISTED HERE!!!)
		# Synchronize two dropdowns
		# Button widgets
		# Button styles
		# Adding a hovertool


# Chapter 4: Putting It All Together! A Case Study

		# Some exploratory plots of the data
		# Beginning with just a plot
		# Enhancing the plot with some shading
		# Adding a slider to vary the year
		# Customizing based on user input
		# Adding dropdowns to the app



		











############################ Chapter 1: Basic plotting with Bokeh


















### A simple scatter plot

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)





### A scatter plot with different shapes

# Plotting multiple datasets on same figure

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)





### Customizing your scatter plots

# The three most important arguments to customize scatter glyphs are color, size, 
# and alpha. Bokeh accepts colors as hexadecimal strings, tuples of RGB values 
# between 0 and 255, and any of the 147 CSS color names. Size values are supplied 
# in screen space units with 100 meaning the size of the entire figure.  The alpha 
# parameter controls transparency. It takes in floating point numbers between 0.0, 
# meaning completely transparent, and 1.0, meaning completely opaque.

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)





### Lines

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date,price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)





### Lines and markers

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date,price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)





### Patches

# In Bokeh, extended geometrical shapes can be plotted by using the patches() glyph function. 
# The patches glyph takes as input a list-of-lists collection of numeric values specifying 
# the vertices in x and y directions of each distinct patch to plot. In this exercise, you 
# will plot the state borders of Arizona, Colorado, New Mexico and Utah. The latitude and 
# longitude vertices for each state have been prepared as lists. Your job is to plot longitude 
# on the x-axis and latitude on the y-axis. The figure object has been created for you as p.

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color='white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)





### Plotting data from NumPy arrays

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0,5,100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x,y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)





### Plotting data from Pandas DataFrames

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)





### The Bokeh ColumnDataSource (continued)

# You can create a ColumnDataSource object directly from a Pandas 
# DataFrame by passing the DataFrame to the class initializer.

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', size=8, source=source, color='color')

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)





### Selection and non-selection glyphs

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time', 
        tools='box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', selection_color='red', nonselection_alpha=0.1,source=source)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)





### Hover glyphs

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)






### ColorMapping

# The final glyph customization we'll practice is using the CategoricalColorMapper 
# to color each glyph by a categorical property.

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)

















############################ Chapter 2: Layouts, Interactions, and Annotations 
















### Creating Rows of Plots

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)

# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)






### Creating columns of plots

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)

# Put plots p1 and p2 in a column: layout
layout = column(p1,p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)





### Nesting rows and columns of plots

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)






### Creating gridded layouts

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1,p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3,p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1,row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)






### Starting tabbed layouts

# Tabbed layouts can be created in Pandas by placing plots or layouts in Panels.
# In this exercise, you'll take the four fertility vs female literacy plots 
# from the last exercise and make a Panel() for each. No figure will be generated 
# in this exercise. Instead, you will use these panels in the next exercise 
# to build and display a tabbed layout.

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')






### Displaying tabbed layouts

# Produces plots with differnt plots accessible through 'tabs' that 
# can be selected above the plot

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)






###  Linked axes

# When displaying multiple plots, we can link the ranges of one plot
# to the ranges of another (x, y, or both).  This links the axes at 
# initialization of the plots and while panning in any of the plots 

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)





### Linked brushing

# By initiating two figures and indicating the same source=source command in each,
# we can link the selections in one plot with the highlighted data in the other.
# The column data source function takes a dataframe as input, and when adding the circle
# glyph we must indicate source=source

# input dataframe:
Country  Continent female literacy fertility   population
0      Chine       ASI            90.5     1.769  1324.655000
1       Inde       ASI            50.8     2.682  1139.964932
2        USA       NAM              99     2.077   304.060000
3  Indonésie       ASI            88.8     2.132   227.345082
4     Brésil       LAT            90.2     1.827   191.971506

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)

# Create row layout of figures p1 and p2: layout
layout = row([p1,p2])

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)






### How to create legends

# note, ColumnDataSource object 'latin_america' and 'africa'  as well as the figures have been provided

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)






### Positioning and styling legends (location & background color)

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)






### Adding a hover tooltip

# note, figure 'p' has been provided
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips=[('Country', '@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)




















############################ Chapter 3: Building interactive apps with Bokeh




















# Using the current document

#Let's get started with building an interactive Bokeh app. This typically begins 
# with importing the curdoc, or "current document", function from bokeh.io. This 
# current document will eventually hold all the plots, controls, and layouts that 
# you create. Your job in this exercise is to use this function to add a single plot 
# to your application.  In the video, Bryan described the process for running a 
# Bokeh app using the bokeh serve command line tool. In this chapter and the one 
# that follows, the DataCamp environment does this for you behind the scenes. Notice 
# that your code is part of a script.py file. When you hit 'Submit Answer', you'll 
# see in the IPython Shell that we call bokeh serve script.py for you.

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure

# Create a new plot: plot
plot = figure()

# Add a line to the plot
plot.line([1,2,3,4,5], [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)






# Add a single slider

# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)

# Create a widgetbox layout: layout
layout = widgetbox(slider)

# Add the layout to the current document
curdoc().add_root(layout)






# Multiple sliders in one document

# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)

# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)






### How to combine Bokeh models into layouts

# Let's begin making a Bokeh application that has a simple slider and plot, 
# that also updates the plot based on the slider. In this exercise, your job 
# is to first explicitly create a ColumnDataSource. You'll then combine a plot 
# and a slider into a single column layout, and add it to the current document.
# After you are done, notice how in the figure you generate, the slider will not 
# actually update the plot, because a widget callback has not been defined. 
# You'll learn how to update the plot using widget callbacks in the next exercise.
# All the necessary modules have been imported for you. The plot is available 
# in the workspace as plot, and the slider is available as slider.

# Create ColumnDataSource: source
source = ColumnDataSource(data={'x':x, 'y':y})

# Add a line to the plot
plot.line('x', 'y', source=source)

# Create a column layout: layout
layout = column(widgetbox(slider), plot)

# Add the layout to the current document
curdoc().add_root(layout)






### Learn about widget callbacks

# You'll now learn how to use widget callbacks to update the state of a Bokeh application, 
# and in turn, the data that is presented to the user. Your job in this exercise is to 
# use the slider's on_change() function to update the plot's data from the previous 
# example. NumPy's sin() function will be used to update the y-axis data of the plot.

# Define a callback function: callback
def callback(attr, old, new):

    # Read the current value of the slider: scale
    scale = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)

    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}

# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)

# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)






### Updating data sources from dropdown callbacks (all code listed here)

# You'll now learn to update the plot's data using a drop down menu instead of a slider. 
# This would allow users to do things like select between different data sources to view.
# The ColumnDataSource source has been created for you along with the plot. Your job 
# in this exercise is to add a drop down menu to update the plot's data.

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)






### Synchronize two dropdowns

# Here, you'll practice using a dropdown callback to update another dropdown's options. 
# This will allow you to customize your applications even further and is a powerful addition 
# to your toolbox.

# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A' 
    if select1 == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)






### Button widgets

# It's time to practice adding buttons to your interactive visualizations. Your job in 
# this exercise is to create a button and use its on_click() method to update a plot.
# All necessary modules have been imported for you. In addition, the ColumnDataSource
# with data x and y as well as the figure have been created for you and are available 
# in the workspace as source and plot.

# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x' : x, 'y': y} 

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)






### Button styles

# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import RadioGroup, CheckboxGroup, Toggle

# Add a Toggle: toggle
toggle = Toggle(button_type='success', label='Toggle button')

# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])

# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))

















########################### Chapter 4: Putting It All Together! A Case Study














### Some exploratory plots of the data

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

# Make the ColumnDataSource: source.  Use only data from 1970
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)






### Beginning with just a plot

# Let's get started on the Gapminder app. Your job is to make the ColumnDataSource object, 
# prepare the plot, and add circles for Life expectancy vs Fertility. You'll also set x 
# and y ranges for the axes.

# As in the previous chapter, the DataCamp environment executes the bokeh serve command 
# to run the app for you. When you hit 'Submit Answer', you'll see in the IPython Shell 
# that bokeh serve script.py gets called to run the app. This is something to keep in 
# mind when you are creating your own interactive visualizations outside of the DataCamp environment.

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
    'pop'     : (data.loc[1970].population / 20000000) + 2,
    'region'  : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700, x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'








### Enhancing the plot with some shading

# Now that you have the base plot ready, you can enhance it by coloring each circle glyph by continent.
# Your job is to make a list of the unique regions from the data frame, prepare a ColorMapper, and add it to the circle glyph.

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'






### Adding a slider to vary the year

# Import the necessary modules
from bokeh.layouts import widgetbox, row
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # set the `yr` name to `slider.value` and `source.data = new_data`
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data = new_data


# Make a slider object: slider
slider = Slider(start=1970, end=2010, step=1,value=1970,title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value',update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)






### Customizing based on user input

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data

    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr

# Make a slider object: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)






### Adding a hovertool

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)
# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)






### Adding dropdowns to the app
