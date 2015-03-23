cdef extern from "spa.h":
	ctypedef struct spa_data:
		int year           
		int month          
		int day            
		int hour           
		int minute         
		double second  
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

		double e0
		double e
		double zenith     
		double azimuth_astro
		double azimuth    
		double incidence  

		double suntransit 
		double sunrise    
		double sunset   

	int spa_calculate(spa_data *spa)
