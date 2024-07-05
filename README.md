
# *Brief Project Overview & Objectives of the Capstone Project*

### *Why we are doing this project ?*

  *1. This Project provides the industry with a well reliable analysis that drive the decision making in the industry of AutoMobiles*

  *2. Capstone projects are generally developed to energize students' critical thinking, problem-solving, oral communication, research, and teamwork abilities*

### *Problem Statement :*

  *1. Description of the Problem :   To predict  the selling prices of AutoMobiles based on the data provided in this analysis*

  *2. Importance of Solving the Problem :  Unveil underlying patterns and contributing factors to the observed price for AutoMobiles based on the data provided ,facilitating more informed decision-making*

  *3. Target Audience or Beneficiaries : General*  

  ### *Objectives and Goals*

  *1. Specific Aims of the Project : Predicting and forecasting prices for AutoMobiles basd on he given data*

  *2. Expected Outcomes : Expected Price*

  ### *Methodology*

  *1. Data Exploration*

   * *A. Data Description :*

     * *The given dataset has the following Features:*
     
       * *Categorical features :  6 Features  [num_of_doors, body_style, drive_wheels,engine_location,num_of_cylinders, fuel_system]*
     
       * *Numerical features : 9 features [price,length, width, height, curb_weight, engine_size,peak_rpm, city_mpg, highway_mpg]*

   * *B. Data Preprocessing :*

     * *Missing Values*

     * *Duplicates Values*

     * *Data types*

     * *Unique values is each column*

     * *Check statistics of data set*

  * *C. Modeling :*

     * *Applied RandomizedSearchCV on liest of models and parameters :*   

          *GradientBoostingRegressor() Model with the following parameters :*
            *SUBSAMPLE: 0.8*
            *N_ESTIMATORS: 100*  
            *MIN_SAMPLES_SPLIT: 2*  
            *MIN_SAMPLES_LEAF: 1*  
            *MAX_DEPTH: 4*  
            *LEARNING_RATE: 0.1*  