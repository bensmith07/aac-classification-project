# Austin Animal Center Outcomes
## Which dogs will be adopted?

### Project Description

This project uses machine learning on data from Austin Animal Center (AAC) in an attempt to predict which dogs will be adopted and which will be transferred. 

### Project Goals

With a reliable and accurate prediction model, Austin Animal Center could better understand how to focus resources on marketing dogs for adoption as well as finding and coordinating transfer partners. This ultimately could lead to more efficient use of city resources and better outcomes for the dogs that find themselves in the city shelter. 

### Initial Questions

- How does the breed of the dog affect it's adoption outcome?

- How does the color of the dog affect it's adoption outcome?

- How does the age of the dog affect it's adoption outcome?

- How does the condition of the dog affect it's adoption outcome?

### Data Dictionary

This data was obtained from the City of Austin's open data portal at data.austintexas.gov. The data is published by the city's Animal Services department and updated on an hourly basis. The data available for this analysis represents all animals taken into the shelter from 10/01/2013 - 03/04/2022.

AAC publishes two separate dataset: "Intakes" and "Outcomes". These datasets were combined to perform this analysis.

From AAC: 
        
        "Intakes represent the status of animals as they arrive at the Animal Center. All animals receive a unique Animal ID during intake."

        "Outcomes represent the status of animals as they leave the Animal Center... Annually over 90% of animals entering the center, are adopted, transferred to rescue or returned to their owners. The Outcomes data set reflects that Austin, TX. is the largest "No Kill" city in the country.

(Note: no additional feature descriptions were included in the data source. Descriptions below are based on my own interpretation of the data.)

### Original Features 
(available from the original dataset with minimal feature engineering)

| Feature | Description |
| ------- | ----------- |
| outcome_type | (target) The method by which the animal left the shelter. Includes, among others, 'Adoption', 'Transfer', 'Return to Owner'. |
| animal_id | a unique identifier for each animal
| intake_type | The method by which the animal arrived at the shelter. Includes, among others, 'Stray', 'Owner Surrender', 'Abandoned' |
| intake_condition | The animal's health condition at the time of intake. Includes, among others, 'Normal', 'Injured', 'Sick', 'Aged' |
| month_intake | The month the animal arrived at the shelter. (Derived from 'datetime_intake') |
| fixed | Whether the animal had been spayed or neutered at the time of intake. (derived from 'sex_upon_intake') |
| sex | The sex of the animal (derived from 'sex_upon_intake')
| age_intake | The estimated age of the animal (in days) at the time of intake. (derived from 'Age upon Intake')
| found_location | The location from which the animal originated before being brought into the shelter. Includes cross streets when available. Sometimes only represents a city or county. Also 'Outside Jurisdiction' which likely represents locations outside Travis County, TX. 


### Engineered Features
(features derived from the original dataset using feature engineering techniques in pandas)
| Feature | Description |
| ------- | ----------- |
| breed_mixed | Whether the animal is a mixed breed (True) or purebred (False). (derived from 'Breed') |
| breed_1 | The primary breed of the animal. The first breed listed in the case of multiple breeds contained in the original 'Breed' column. |
| color_1 | The primary color of the animal. The first color listed in the case of multiple colors contained in the original 'Color' column. |
| n_previous_stays | The number of previous times this animal has been brought to the shelter. Derived from duplicate animal id's with separate datetime_intake entries in the original data. |
| stay_id | a unique identifer for each time a particular animal was brought into the shelter. Derived from 'animal_id' and 'n_previous_stays' |
| found_in_austin | Whether the animals found location was within Austin city limits. derived from 'found_location' |
| found_in_travis | Whether the animal was found in a location that is outside city limits but within Travis County. derived from 'found_location' |
| found_other | Whether the animal was found inside the city limits of a city other than Austin. derived from 'found_location' |
| is_pitbull | Whether the animal's primary breed is of a pit bull-type. derived from 'breed_1' |
| is_black | Whether the animal's primary color is black. derived from 'color_1' |
| akc_breed_group | Represents the breed group to which the animal's primary breed belongs, as defined by the American Kennel Club. Derived from 'breed_1'. Source: https://www.akc.org/public-education/resources/general-tips-information/dog-breeds-sorted-groups/ |
