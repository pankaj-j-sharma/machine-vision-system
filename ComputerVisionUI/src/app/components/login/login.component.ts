import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';


@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {

  processing: Boolean = false;
  error: Boolean = false;

  constructor(
    private restApiService: RestApiService,
    private router: Router,
  ) {}

  ngOnInit() {
    if (this.restApiService.hasLoginToken()) {
      //this.handleLoginSuccess();
    } 
  }

  validate(data:any){
    console.log('data',data);
    this.error  = false;
    this.processing  = true;
    this.restApiService.validateUserLogin(data).subscribe(
      resp => {

        this.processing = false;
        this.error  = false;

        if (resp.results) {
          this.restApiService.saveLoginToken(resp.results);
          this.handleLoginSuccess();
        } else {
          this.error  = true;
        }
      });
  }

  private handleLoginSuccess() {
    this.processing = false;
    this.error  = false;
    this.router.navigate(['/home']);
  }


}
