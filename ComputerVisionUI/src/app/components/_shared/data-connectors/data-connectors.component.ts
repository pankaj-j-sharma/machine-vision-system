import { AfterViewInit, Component, ElementRef, OnInit, SimpleChanges, ViewChild } from '@angular/core';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

@Component({
  selector: 'app-data-connectors',
  templateUrl: './data-connectors.component.html',
  styleUrls: ['./data-connectors.component.scss']
})
export class DataConnectorsComponent implements OnInit ,AfterViewInit {

  selectedConnector:any={};
  selectedSubOption:any={};
  subOptions:any[]=[];
  allSubOptns:any[]=[];
  allConnectors : any[]=[];
  allConnectorsMaster : any[]=[];
  formData = new FormData();
  processing:boolean = false;
  isOptionSelected:boolean = false;
  modalHeading:string="Connectors";
  loadedDirectories:any[]=[];
  isDirectoriesLoaded:boolean=false;
  currentDirectoryPath:any[]=[];
  pathSep:string="\\";
  currentPathBreadcrumb:string="";
  assetPath:string='../../../../assets/images/common/data-connectors/';
  credentialsFile:File[]=[];
  loadConnectors: boolean= false;

  @ViewChild('myconnectors') myconnectors:any;
  
  constructor(private restAPIService : RestApiService) { }

  ngOnInit(): void {
  }

  ngAfterViewInit() {
    if(this.loadConnectors){
      this.loadAllMetaData();    
    }
  }

  ngOnChanges(changes: SimpleChanges) {  
    console.log('changes in connector',changes);
    // this.generateCharts(this.appChart);
  }

  toggleLoadConnectors(){
    this.loadConnectors=true;
    this.loadAllMetaData();    
  }

  openTab(name:any){
    this.selectedConnector = this.allConnectors.find(c=>c.name ==name);
    this.subOptions = this.selectedConnector.subOptions;
    if(name.toUpperCase()=='ALL'){
      this.subOptions=[];
      this.allConnectors.forEach(c=> {
        c.subOptions.forEach((s:any) => {
          this.subOptions.push(s);
        })
      });
      this.subOptions = this.subOptions.sort((a:any,b:any)=>{
        if (a.name < b.name)
        return -1;
        if (a.name > b.name)
          return 1;
        return 0;
      });
      this.allSubOptns = this.subOptions;
    }
  }

  openSubOption(name:any){
    this.selectedSubOption = this.subOptions.find(f=> f.name==name);
    this.isOptionSelected = true;
    this.modalHeading=this.selectedSubOption.name;
  }

  search(event:any){
    let keyword:string = event.target.value;
    console.log('searched ',keyword);
    if(keyword && keyword !=""){
      // this.subOptions = this.allSubOptns.filter(s => s.name.toUpperCase().indexOf(keyword.toUpperCase()) >-1)
      // this.allConnectors.forEach(con => {
      //   con.subOptions.forEach((s:any)=> s.name.toUpperCase().indexOf(keyword.toUpperCase()) >-1);
      //   console.log('con',con.subOptions);
      // })      
    }
  }

  loadAllMetaData(){
    this.loadAllConnectors();
  }

  loadAllConnectors(){
    this.processing=true;
    this.restAPIService.getAllDataConnectors(this.formData).subscribe(resp=> {
      if('success' in resp){
        let results = resp.results;
        this.allConnectors = results;
        this.openTab('All');        
        console.log('connector result',this.allConnectors);
      }
      this.processing= false;
    })
    this.allConnectorsMaster = this.allConnectors;
  }

  loadConnectorData(data:any){
    console.log('data',data);

    this.currentPathBreadcrumb = "";
    this.currentDirectoryPath=[]; // clear the breadcrumb path
    this.loadedDirectories = [];

    this.formData= new FormData();
    this.formData.append('connector_name',this.selectedSubOption.name);
    let userId = this.restAPIService.getCurrentUserId();
    if(userId){
      this.formData.append('userId',userId.toString());
    }

    // form data post as per specific to a connector
    if(this.modalHeading == 'Amazon S3 Bucket'){
      this.pathSep="\\";
      this.formData.append('access_key_id',data.accesskeyid);
      this.formData.append('access_key_secret',data.accesskeysecret);
      this.formData.append('bucket_name',data.bucketname);      
    }
    else if(this.modalHeading == 'Azure Blob Storage'){
      this.pathSep="/";
      this.formData.append('storage_account_name',data.storageaccountname);
      this.formData.append('storage_account_key',data.storageaccountkey);
      this.formData.append('container_name',data.containername);      
    }
    else if(this.modalHeading == 'Azure Data Lake Storage Gen2'){
      this.pathSep="/";
      this.formData.append('storage_account_name',data.storageaccountname);
      this.formData.append('storage_account_key',data.storageaccountkey);
      this.formData.append('filesystem_name',data.containername);      
    }

    else if(this.modalHeading == 'Google BigQuery' || this.modalHeading == 'Google Cloud Storage' || this.modalHeading == 'Google Cloud Datastore'){
      this.pathSep="/";
      this.formData.append('credential_file',this.credentialsFile[0]);
    }

    else if(this.modalHeading == 'Snowflake'){
      this.pathSep=">";
      this.formData.append('username',data.snowflakeusername);
      this.formData.append('password',data.snowflakepassword);
      this.formData.append('account_name',data.snowflakeaccountname);      
    }
    else if(this.modalHeading == 'PostgreSQL Database'){
      this.pathSep=">";
      this.formData.append('hostname',data.postgreshostname);      
      this.formData.append('portno',data.postgresportno);      
      this.formData.append('username',data.postgresusername);
      this.formData.append('password',data.postgrespassword);
      this.formData.append('database_name',data.postgresdbname);      
    }

    
    this.isDirectoriesLoaded=true;
    this.processing=true;
    this.restAPIService.loadConnectorDirectories(this.formData).subscribe(resp=> {
      if('success' in resp){
        let results = resp.results;
        console.log('connector result',results);
        if(this.modalHeading == 'Snowflake'){
          this.loadedDirectories = results.all_dwh;
        }else if(this.modalHeading == 'PostgreSQL Database' || this.modalHeading == 'Google BigQuery' ){
          this.loadedDirectories = results.all_db;
        }else if(this.modalHeading == 'Google Cloud Datastore' ){
          this.loadedDirectories = results.all_tbl;
        }        
        else{
          this.loadedDirectories = results.all_directories && results.all_directories.length>0 ? results.all_directories : results.all_buckets;
        }
      }
      this.processing= false;
    })

  }

  navigateNext(data:any){
    if (this.modalHeading =='Snowflake'){
      this.loadDbObjects(data);
    }else if(this.modalHeading=='PostgreSQL Database'){
      this.loadDbObjects(data);      
    }else if(this.modalHeading =='Google BigQuery'){
      this.loadDbObjects(data);
    }else if(this.modalHeading =='Google Cloud Datastore'){
    }else{
      this.openDirectory(data);
    }
  }

  loadGCPContent(data:any){

  }

  onFileChange(event:any){
    this.credentialsFile= event.target.files;
  }

  loadCredentialFile(data:any){

  }

  loadDbObjects(dbObject:any){
    let prefixArr:any[]=[];
    if(this.modalHeading=='Snowflake'){
      prefixArr = ['warehouse_name','database_name'];
    }else if(this.modalHeading=='PostgreSQL Database' || this.modalHeading=='MySQL Database'){
      prefixArr = ['database_name'];
    }else if(this.modalHeading=='Oracle Database'){
      prefixArr = ['sid'];
    }else if (this.modalHeading=='Google BigQuery'){
      prefixArr=['dataset_id'];
    }

    if(dbObject =='..'){
      this.formData.delete(prefixArr[this.currentDirectoryPath.length-1]);
      this.currentDirectoryPath.pop();      
    }else{
      this.currentDirectoryPath.push(dbObject);
      if (this.currentDirectoryPath){
        this.formData.delete(prefixArr[this.currentDirectoryPath.length-1]);
        this.formData.append(prefixArr[this.currentDirectoryPath.length-1],dbObject);      
      }      
    }

    this.processing=true;
    this.restAPIService.loadConnectorDirectories(this.formData).subscribe(resp=> {
      if('success' in resp){
        let results = resp.results;
        console.log('connector result',results,results.all_tbl);
        this.loadedDirectories = results.all_tbl;
        if(this.loadedDirectories.length ==0){
          this.loadedDirectories = results.all_db;
          if(this.loadedDirectories.length ==0){
            this.loadedDirectories = results.all_dwh;
          }
        }else{
          this.loadedDirectories = this.loadedDirectories.map(f=>f.name);
        }
        this.currentPathBreadcrumb = this.currentDirectoryPath.join(this.pathSep);
      }
      this.processing= false;
    });
  }

  openDirectory(prefix:any){
    if(prefix =='..'){
      this.currentDirectoryPath.pop();
    }else{
      this.currentDirectoryPath.push(prefix);
    }
    let finalPrefix = this.currentDirectoryPath.join(this.pathSep);
    if(this.modalHeading=='Google Cloud Storage'){
      this.formData.delete('bucket_name');
      this.formData.append('bucket_name',finalPrefix.split(this.pathSep)[0]);    
    }else{
      this.formData.delete('prefix');
      this.formData.append('prefix',finalPrefix);    
    }
    this.processing=true;
    this.restAPIService.loadConnectorDirectories(this.formData).subscribe(resp=> {
      if('success' in resp){
        let results = resp.results;
        console.log('connector result',results);
        this.loadedDirectories = results.all_buckets ? results.all_buckets :(results.all_directories && results.all_directories.length>0 ? results.all_directories: results.all_files);
        this.currentPathBreadcrumb = this.currentDirectoryPath.join('/');
      }
      this.processing= false;
    });
  }  
}
