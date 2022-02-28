import { Component, OnInit } from '@angular/core';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';
import { Clipboard } from '@angular/cdk/clipboard';


@Component({
  selector: 'app-api-endpoints',
  templateUrl: './api-endpoints.component.html',
  styleUrls: ['./api-endpoints.component.scss']
})
export class ApiEndpointsComponent implements OnInit {

  processing:boolean = false;
  selectedModel : string = "";
  selectedModelName : string = "";  
  availableModels : any[] = [];
  apiKeyGenerated=false;
  genkey:string="9f3aFtypJHMdDYNNFFd_3YZVeB8eQdPQXZhde73ibW82zPnHFzZCosKPmh5ZcMuj2bn9PxHC8hF9jXyK";
  formData = new FormData();
  apiEndpointUrl = "http://visimatic.tatatechnologies.com/api/v1/predict/"
  apiUploadFileUrl = "http://visimatic.tatatechnologies.com/api/v1/upload/";
  apiUsername = "";
  generatedOn = "";
  genButtonText = "Generate API key";

  constructor(private restAPIService : RestApiService ,  private clipboard: Clipboard) { }

  ngOnInit(): void {
    this.loadMetaData();
  }

  selectAvlModel(event:any){
    event.preventDefault();
    this.selectedModel = event.target.value;
    this.selectedModelName = this.availableModels.find(m=> m.id==this.selectedModel).name;
    this.loadUserApiKeys();
  }

  loadMetaData(){
    this.processing=true;
    this.restAPIService.getAllAvailableModels({'UserId':this.restAPIService.getCurrentUserId(),'AppName':this.restAPIService.getCurrentSelectedApp()}).subscribe(resp=>{
        if ('success' in resp){
          let results = resp.results;
          for (let i = 0; i < results.length; i++) {
            this.availableModels.push(results[i]);          
          }
        }
        this.processing=false;
      });
    }

  generateAPIKeys(){
    this.processing = true;
    this.formData = new FormData();
    let userid = this.restAPIService.getCurrentUserId();
    if(userid){
      this.formData.append('userId',userid.toString());
    }
    this.formData.append('key_length','60');
    this.formData.append('modelId',this.selectedModel);
    this.restAPIService.generateUserApiKey(this.formData).subscribe(resp=>{
      if ('success' in resp){
        let results = resp.results;
        this.genkey = results.api_key;
        this.apiKeyGenerated = true;
      }
      this.processing=false;
    });
  }   
  
  loadUserApiKeys(){
    this.processing = true;
    this.formData = new FormData();
    let userid = this.restAPIService.getCurrentUserId();
    if(userid){
      this.formData.append('userId',userid.toString());
    }
    this.formData.append('modelId',this.selectedModel);
    this.restAPIService.loadUserApiKey(this.formData).subscribe(resp=>{
      if ('success' in resp && resp.results != null){
        let results = resp.results;
        this.genkey = results.api_key;
        this.apiUsername = results.username;
        this.generatedOn = results.created_on;
        this.genButtonText="Regenerate API key";
        this.apiKeyGenerated = true;
      }else{
        this.genkey = "";
        this.apiUsername = "";
        this.generatedOn = "";
        this.apiKeyGenerated = false;
        this.genButtonText="Generate API key";
      }
      this.processing=false;
    });
  }

  copyToClipBoard(inputElement:any){
    inputElement.select();
    inputElement.setSelectionRange(0, 0);  
    this.clipboard.copy(inputElement.value);    
  }  
}
