import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

export interface UserInfo {
  FirstName:string,
  LastName:string,
  UserName:string,
  Password:string,
  Organization:string,
  EmailAddress:string,
  ContactNumber:string,
  ImgSrc:string
}

@Component({
  selector: 'app-user-profile',
  templateUrl: './user-profile.component.html',
  styleUrls: ['./user-profile.component.scss']
})
export class UserProfileComponent implements OnInit {

  processing:boolean = false;
  userId:any = this.restApiService.getCurrentUserId();
  uploadedFiles:File[]=[];
  formData = new FormData();
  userInfo: UserInfo = {
    FirstName:'',
    LastName:'',
    UserName:'',
    Password:'',
    Organization:'',
    EmailAddress:'',
    ContactNumber:'',
    ImgSrc:'../../../assets/images/common/default-profile-picture.png'
  };

  constructor(
    private router: Router,
    private restApiService: RestApiService    
  ) { }

  ngOnInit(): void {
    this.loadUserProfile();
  }

  loadUserProfile(){
    this.processing=true;
    this.userId = this.userId?this.userId:'-1';
    if (this.userId)
    {
      this.formData = new FormData();
      this.formData.append('userid',this.userId.toString());    
      this.restApiService.getUserDataFromAPI(this.formData).subscribe(resp=> {
        if('success' in resp){
          let results = resp['results'];
          console.log('results',results);
          this.userInfo.EmailAddress = results.emailid;
          this.userInfo.FirstName = results.firstname;
          this.userInfo.LastName = results.lastname;
          this.userInfo.ContactNumber = results.mobile;
          this.userInfo.Organization = results.organization;
          this.userInfo.UserName = results.username; 
          this.userInfo.Password = results.passwd; 
          
          if(results.profileimg){
            this.userInfo.ImgSrc = this.restApiService.getAllSavedFiles(results.profileimg);
          }         
        }
        this.processing=false;
      });
    }
  }

  saveUserData(data:any){
    this.processing=true;
    console.log('data',data);
    
    this.formData = new FormData();
    this.formData.append('userid',this.userId.toString());    
    this.formData.append('mobile',data.ContactNumber.toString());    
    this.formData.append('emailid',data.EmailAddress);    
    this.formData.append('firstname',data.FirstName);    
    this.formData.append('lastname',data.LastName);    
    this.formData.append('organization',data.Organization);    
    this.formData.append('passwd',data.Password);    
    this.formData.append('username',data.UserName);    
    
    for (let iter = 0; iter < this.uploadedFiles.length; iter++) {
      const fileElement = this.uploadedFiles[iter];    
      this.formData.append(fileElement.name,fileElement);  
    }
      

    this.restApiService.saveUserInfoData(this.formData).subscribe(resp=> {
      if('success' in resp){
        let results = resp['results'];
        console.log('results',results);
        this.userInfo.EmailAddress = results.emailid;
        this.userInfo.FirstName = results.firstname;
        this.userInfo.LastName = results.lastname;
        this.userInfo.ContactNumber = results.mobile;
        this.userInfo.Organization = results.organization;
        this.userInfo.UserName = results.username; 
        this.userInfo.Password = results.passwd; 
        
        if(results.profileimg){
          this.userInfo.ImgSrc = this.restApiService.getAllSavedFiles(results.profileimg);
        }         
      }
      this.processing=false;
    });
}

  onFileChange(event:any){
    let files = event.target.files;
    this.uploadedFiles=files;
    console.log('files',files,files.length);
    if(files.length>0){
      var reader = new FileReader();
      reader.onload = (e: any) => {
        this.userInfo.ImgSrc = e.target.result;
      };    
      reader.readAsDataURL(files[0]);        
    }
  }

  navigateBack(){
    let back = this.restApiService.getItemFromStorage('current');
    if(back){
      this.router.navigate([back]);
    }
  }
}
