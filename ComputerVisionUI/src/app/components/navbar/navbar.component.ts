import { AfterViewInit, Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

declare var $: any;

export interface NavSideBarItem{
  title:string
  description:string
  iconClass:string
  url:string
  roles:string[]
  subItems:NavSideBarItem[]
}

export interface AppAlertItem{
  title:string
  url:string
  titleClass:string
  titleIcon:string
  alertTime:string
  description: string
}

export interface AppMsgItem{
  name:string
  url:string
  msgTime:string
  data:string
}


@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.scss']
})
export class NavbarComponent implements OnInit,AfterViewInit {

  navSideBarItems:NavSideBarItem[]=[];
  allAppAlerts: AppAlertItem[]=[];
  allAppMsgs: AppMsgItem[]=[];
  tmpData= {} // need to decide on the data to pass
  currentUserRole="";
  userFirstName:string="";
  userOrganisation:string="";
  connectIndicator:string = "text-muted";
  formData = new FormData();

  userProfileImg:any;
  userId:any = this.restApiService.getCurrentUserId();

  constructor(
    private router: Router,
    private restApiService: RestApiService
  ) { }

  ngAfterViewInit(): void {
    // this loads after dom load 
    console.log('viewafter init');
    this.onLoadJquery();
  }

  ngOnInit(): void {
    this.restApiService.saveItemToStorage('current',window.location.pathname);
    this.tmpData = {'UserId':this.restApiService.getCurrentUserId()};
    this.loadNavItemMetaData();
    this.loadAppAlertData();
    this.loadAppUserMessages();
    this.loadUserProfile();
  }

  loadUserProfile(){
    this.userProfileImg = this.restApiService.getItemFromStorage('profilepic');
    if(!this.userProfileImg){
      this.userId = this.userId?this.userId:'-1';
      if (this.userId)
      {
        this.formData = new FormData();
        this.formData.append('userid',this.userId.toString()); 
        this.restApiService.getUserDataFromAPI(this.formData).subscribe(resp=> {
          if('success' in resp){
            let results = resp['results'];
            if(results.profileimg){
              this.userProfileImg = this.restApiService.getAllSavedFiles(results.profileimg);
              this.restApiService.saveItemToStorage('profilepic',this.userProfileImg);
            }           
          }
        });  
      }  
      this.userProfileImg = "../../../assets/images/common/default-profile-picture.png";
    }
  }

  loadNavItemMetaData(){
    this.currentUserRole = this.restApiService.getCurrentUserData().role||"null";
    this.restApiService.getMenuItemOptions(this.currentUserRole).subscribe(resp => {
      let urlPath = window.location.pathname;
      this.navSideBarItems= resp.results; 
      // show only home menu for below options
      if (urlPath.endsWith('tracking')){
        this.navSideBarItems = this.navSideBarItems.filter(n=>n.title=='Home' || n.url.endsWith('processtracking'));
      }
      else{
        this.navSideBarItems = this.navSideBarItems.filter(n=>!n.url.endsWith('processtracking'));        
      }
    })
  }

  loadAppAlertData(){
    this.restApiService.getAppAlertItems(this.tmpData).subscribe(resp => {
      if ('success' in resp){
        let results = resp.results;
        for (let i = 0; i < results.length; i++) {
          this.allAppAlerts.push({
            title:results[i].Title,
            description:results[i].Description,
            alertTime:results[i].AlertTime,
            titleClass:results[i].TitleClass,
            titleIcon:results[i].TitleIcon,
            url:results[i].Url
          });
        }
      }
    })
  }

  loadAppUserMessages(){
    this.restApiService.getAppUserMsgItems(this.tmpData).subscribe(result => {
      this.allAppMsgs= result.allMsgs;
    })
  }

  showUserProfile(){
    let userUrl = window.location.pathname.split("-")[0]+"-profile";
    this.router.navigate([userUrl]);
  }

  logoutUser(){
    this.router.navigate(['login']);
  }

  onLoadJquery(){
    this.toggleSidebar();
    //$("body").toggleClass("sidenav-toggled");
    // $.ready not working on main.js as jQuery loads before css hence added here tooltip
    $('.navbar-sidenav [data-toggle="tooltip"]').tooltip({
        template: '<div class="tooltip navbar-sidenav-tooltip" role="tooltip" style="pointer-events: none;"><div class="arrow"></div><div class="tooltip-inner"></div></div>'
    });

    // Force the toggled class to be removed when a collapsible nav link is clicked
    $(".navbar-sidenav .nav-link-collapse").on('click', function(e:any) {
        e.preventDefault();
        $("body").removeClass("sidenav-toggled");
    });
  }

	toggleSidebar() {
    $("body").toggleClass("sidenav-toggled",1);
    $(".navbar-sidenav .nav-link-collapse").removeClass("collapsed");
    $(".navbar-sidenav .sidenav-second-level, .navbar-sidenav .sidenav-third-level, .menu-icon").removeClass("show");
    // added for hiding al together left menu
    $('.sidenav-toggler').animate({'height':'toggle'},1);    
    $('#exampleAccordion').animate({'height':'toggle'},1);
}  

}
