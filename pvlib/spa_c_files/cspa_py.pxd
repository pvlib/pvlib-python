cdef extern from "spa.h":
	ctypedef struct spa_data:
		int year           
		int month          
		int day            
		int hour           
		int minute         
		int second  
		double delta_ut1   
		double delta_t     
		double timezone    
		double longitude   
		double latitude        

		double elevation       

		double pressure        

		double temperature     

		double slope           

		double azm_rotation        

		double atmos_refract    

		int function       

		double zenith     
		double azimuth180 
		double azimuth    
		double incidence  

		double suntransit 
		double sunrise    
		double sunset   

	int spa_calculate(spa_data *spa)
