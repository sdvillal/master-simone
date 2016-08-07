# coding=utf-8
"""
Simone's email
--------------------------------------------------------------------------------

Thanks for accepting to have a look at the data for me.
I attach the following (all work in progress docs at the moment
and I will have to revise the script to read better and for referencing etc.,
but hope that it's enough to give you the context):

1. Explanation of the method to get an idea of what I did
   including a full set of stimuli and their details.
2. The excel file, extracting what I think is important data
   and the very basic 'analysis' I have done so far.

I used the staircase method for this experiment which I have tried to explain
in my methods section. Hope you can follow.

The excel file has 4 tabs:

1- 'Results lum lab'
  Is where the data I got from the experiment is.
  Here are also the staircase models for each participant and my attempt
  at finding variation in responses between participants.
  The staircase graphs are not as expected as the last 3 reversals are
  not always in the same zone so I might have to compare averaging the
  last 2 only and compare the results with averaging 3 or even omit
  very strange staircases from the data analysis altogether.
  For variation in response between participants I have arranged the results
  per stimulus and where I had more than 11 responses (I chose the number arbitrarily)
  I plotted the percentage of responses that answered 'chromatic' is brighter and
  those that responded 'achromatic is brighter' to find inconsistencies.

2- 'Analysis col'
  Is where I find the difference in brightness between chromatic and achromatic
  for each participant. This is done by deducting Luminance of chromatic
  from luminance of achromatic at threshold. Luminance of achromatic at threshold
  is the average of the last 3 reversals in the staircase. In this sheet is also
  a measure of the bias towards first or second stimulus for each particpant.
  I did this by taking the difference in results for staircases with stimuli
  shown in different orders

3- 'INTRA-P VAR col'
  Here I tried to find variation within each participant and how it varies for
  the different stimuli.

4- 'TABLES'
  A summary of the results linked back to the previous three sheets.

What I would like to assess is the following:

 a. If the change in perceived brightness between chromatic and achromatic follows
    normality and if it is significant based on the hypothesis
    'the room with a chromatic surface is perceived brighter
    than a room with an achromatic surface of the same luminance'.
    It seems that it is not but I have to show it statistically.

 b.  the difference in results between males and females

 c.  the difference in results between lighting/architecture/interior
     design participants and others

 d.  the difference in results for the three different colours

 e.  the difference in results between those staircases where a chromatic
     stimulus was shown first and where an achromatic stimulus was shown first.


 Note: a) is the most important and the others are secondary,
          even if bias is worth mentioning as the results are not consistent
          (This is perception after all!!) and I would like to show this)


Since I did the experiment I have spoken to a researcher in the department
who has used this method and she highlighted a way that I could have improved
it so I am carrying out the experiment with three returning participants
and with the amended method on Tuesday. The results of a pilot of this change
in method that I did with Carlos is included under participant '0000'.
Basically what the revision involves is that the experimenter only
changes direction if a reversal (change in opinion) is repeated 2 times.
When I have these results from these 3 participants, I will have to compare
the results of the 2 methods for the 3 participants and comment about both methods.

If it easy for you to suggest a test that would be good to analyse the above
then please let me know and I can try to work it out. And if you think that
there might be something else worth analysing it will also be great help for me.

Sorry for the long winded e-mail - I am reading what I wrote to you and really
think that it may be difficult for you to follow. If you think that it would be
quicker to have a skype chat, let me know.
--------------------------------------------------------------------------------
"""

from .munging import staircases, MY_DIR, DATA_DIR
