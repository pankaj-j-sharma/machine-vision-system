import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

declare var jQuery:any;

export interface HomePageApps{
  Name:string
  Description: string
  Url:string
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})

export class HomeComponent implements OnInit {

  isAppSelected:boolean = false;
  selectedAppCode:string="";
  selectedRoleCode:string="";

  arrOfRows:number[]=[];
  numberOfCols:number=4;
  homePageApps: HomePageApps[] = [];

  constructor(private router: Router,
    private restApiService: RestApiService    
    ) { }

  ngOnInit(): void {
    this.loadAppMetaData();
  }

  appSelection(code:string,nav:string){
    this.isAppSelected = true;
    this.selectedAppCode = code;
    if(nav=='norole'){
      this.restApiService.saveCurrentUserData({'role':'bu','app':this.selectedAppCode});   
      this.router.navigate([this.selectedAppCode]);
    }
  }

  roleSelection(code:string){
    this.selectedRoleCode = code;
    jQuery("#roleModal").modal("hide");
    this.restApiService.saveCurrentUserData({'role':this.selectedRoleCode , 'app':this.selectedAppCode});   
    let navigateUrl = "";
    if(this.selectedAppCode == 'processtracking'){
      navigateUrl = this.selectedRoleCode+'-process-tracking' ;
    }else{
      navigateUrl = this.selectedRoleCode+'-'+this.restApiService.getDefaultPage() ;
    }

    console.log('navgate to',navigateUrl);
    this.router.navigate([navigateUrl]);
  }  

  loadAppMetaData(){
    let userData={role:'DS'}
    this.restApiService.getHomePageApps(userData).subscribe(resp=> {
      this.homePageApps = resp.results;
      this.homePageApps = this.homePageApps.filter(app=> app.Name =='defectdetection' || app.Name=='processtracking'||app.Name=='safetycompliance');
      console.log(this.homePageApps);
      this.arrOfRows=Array.from({length:Math.floor(this.homePageApps.length/this.numberOfCols)+1},(x,i) => i );
    })
  }

}
