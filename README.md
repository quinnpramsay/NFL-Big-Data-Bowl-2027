# NFL-Big-Data-Bowl-2027

Other competitors have found that the data is fairly unreliable

It is only tracking data from the combine, no college game data, so the hypothesis and predicted value is more important than anything else. 

Based on preliminary findings from myself and other people DB back-pedal and transition, DL and OL get-off or first step, WR route breaks are the only viable projects. 

As 30% of the project is graded on “Football” I believe this is in regards to the predictive value. It is a very small project that you are not expected to draw major findings from. But building a good algorithm or statistic from this messy large dataset to build success 
Player_career_successes is useless, with only 2023-2025 players, season outcomes don’t exactly matter and have way too little context. 

With such small causality things like lack of playing time or team they were drafted to can heavily affect the outcomes of this. 

For defensive backs there is almost no outcome metric at all

It is only 510 players, divided down to one position group, and that having players may not even get any snaps over the small 3 year sample size, It is much more about hypothesis than actual model. 

- Based on such a small set of data I am ore inclined to use more regularized regression and gradient boosting model instead of neural networks. 

# Previous Winners/Qualifiers

Looking back on previous Big Data bowl completitions the one large trend that stands out is the people used mathematical norms to create their own new metrics within the sport instead of just building a machine learning pipeline. They also used context specific measurements, ex. 100 out of 200 tackles versus 40 out of 50 tackles. 

Ideas based on that:

- Can we identify which Combine movement characteristics translate into effective movement in the trenches, where athletic ability must be applied under physical constraints?
- Does Combine-measured lateral agility and acceleration predict a defensive lineman's ability to recover and generate pressure after his initial pass-rush movement is disrupted?
- Coverage Recovery Index: How well a DB recovers from a break in a wide receivers route and how well that can be derived from combine data in reaction time - the biggest hurdle will be identifying route breaks in NFL games. 
- Can Combine movement data identify tight ends who are effective in both receiving and blocking situations?

# Data
Uses kaggle api call to import all the data from kaggle. must pip install --upgrade kagglehub in terminal before running data.py script. 
