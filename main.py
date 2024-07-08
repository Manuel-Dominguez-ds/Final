from Trainer import *
from Scorer import *

if __name__=='__main__':
    option=int(input('Chose an option: \n1. Train the model\n2. Score the model\n3. Train and Score the model\n\nPress Enter to continue...'))
    while option not in [1,2,3]:
        option=int(input('Invalid option. Please enter a valid option: \n1. Train the model\n2. Score the model\n3. Train and Score the model\n\nPress Enter to continue...'))
    if option==1:
        trainer = Trainer('Data/online_shoppers_intention.csv')
        trainer.orchestrator()
    elif option==2:
        scorer=Scorer('Data/test.csv')
        scorer.orchestrator()
    elif option==3:
        trainer = Trainer('Data/online_shoppers_intention.csv')
        trainer.orchestrator()
        scorer=Scorer('Data/test.csv')
        scorer.orchestrator()
    print('For MLFlow UI, run the following command in the terminal and then follow the link: \nmlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db')
    print('Thank you for using the application!')    
    