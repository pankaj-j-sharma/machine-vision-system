import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RestApiService {

  // private BASE_URL = "http://127.0.0.1:5002/";
  //private BASE_URL = "http://172.28.138.198/lflask/";

  /* take the hostname dynamically from the window object*/
  private BASE_URL = window.location.hostname=="localhost" ? "http://127.0.0.1:5002/" : "http://"+window.location.hostname+"/lflask/";

  private userLoginUrl = this.BASE_URL+"login";
  private generateUserApiKeyDataUrl = this.BASE_URL+"generateapikey";
  private loadUserApiKeyDataUrl = this.BASE_URL+"loaduserapikey";
  private userProfileDataUrl = this.BASE_URL+"userprofile";
  private saveuserProfileUrl = this.BASE_URL+"saveuserprofile";
  
  private getMenuItemsUrl = this.BASE_URL+"getmenuitem";
  private getHomePageItemsUrl = this.BASE_URL+"gethomeapps";
  private getAppAlertItemsUrl = this.BASE_URL+"getappalerts";
  private getUserMsgItemsUrl = this.BASE_URL+"getusermsgs";

  private defectDetectionDataUrl = this.BASE_URL+"detectsteeldefects";
  private surfaceDefectDetectionDataUrl = this.BASE_URL+"detectsurfacedefects";
  private metalCastDefectDetectionDataUrl = this.BASE_URL+"detectmetalcastdefects";
  private hardHatDetectionDataUrl = this.BASE_URL+"detecthardhatpresent";
  private steelDefectDetectionDataUrl = this.BASE_URL+"detectsteeldefects";
  private packageDamageDetectionDataUrl = this.BASE_URL+"packagedamagedetection";
  private imageAugmentationDataUrl = this.BASE_URL+"augmentimages"; 

  private getSavedFileDataUrl = this.BASE_URL+"getfile"; 
  private getS3FileDataUrl = this.BASE_URL+"getS3Url"; 
   
  private getAllSavedFileDataUrl = this.BASE_URL+"getallfile";  

  private createProjectDataUrl = this.BASE_URL+"createproject";
  private getProjectDataUrl = this.BASE_URL+"getallprojects";
  private getProjectMetaDataUrl = this.BASE_URL+"getprojectmetadata";
  private saveProjectStepDataUrl = this.BASE_URL+"saveprojectstepdata";
  private savePredictionFilesDataUrl = this.BASE_URL + "savepredictionfiles";
  private saveFilesDataForCustomTrainUrl = this.BASE_URL+"savefilesforcustomtrain";
  private loadDraftCustomTrainDataUrl = this.BASE_URL + "loaddraftcustomtrain";
  private loadDraftCustomLabelDataUrl = this.BASE_URL + "loaddraftcustomlabel";
  private loadDraftPredictionDataUrl = this.BASE_URL + "loaddraftpreddata";
  private runCustomTrainingUrl = this.BASE_URL + "runcustomtrain";
  private loadProjectStepsDataUrl = this.BASE_URL+"loadprojectstepsdata";

  private getAllAvailableModelDataUrl = this.BASE_URL+"getallavlmodels";
  private loadPredictionHistoryDataUrl = this.BASE_URL+"loadpredictionhistory";
  private removePredictionHistoryDataUrl = this.BASE_URL+"removepredictionhistory";
  private saveTrainTagsforProjectUrl = this.BASE_URL+"savetraintagsforproject";
  private runTrainingForProjectUrl = this.BASE_URL+"runtrainingforproject";
  private runInferencingForProjectUrl = this.BASE_URL+ "runinferencingforproject";
  private deleteProjectImagesUrl = this.BASE_URL+"deleteprojectimages";
  private deleteProjectItersUrl = this.BASE_URL+"deleteprojectiterations";
  private deleteProjectUrl = this.BASE_URL+"deleteproject";
  private loadIterationsDataUrl = this.BASE_URL+"loadalliterations";
  private loadAllProcessVideosDataUrl = this.BASE_URL+"loadallprocessvideos";
  private saveCustomLabelsForImgUrl = this.BASE_URL + "savecustomlabels";  
  private saveCustomLabelDataUrl = this.BASE_URL + "savefilesforcustomlabelling";
  private saveAugFilesDataUrl = this.BASE_URL + "savefilesforaugmentation";
  private loadAllAugResourcesUrl = this.BASE_URL + "loadallresourcesaug";
  private loadCustomLabelFileDataUrl = this.BASE_URL + "loadcustomlabeldata";
  private loadAugmentationFilesDataUrl = this.BASE_URL + "loadaugmentationdata";
  private saveAugmentationResLabelUrl = this.BASE_URL + "saveaugresultlabel";
  private loadAllResourcesUrl = this.BASE_URL + "loadallresourcesurl";
  private inferenceForUserModelUrl = this.BASE_URL + "inferenceforusermodel";

  private getAllDataConnectorsUrl = this.BASE_URL+"getalldataconnectors";
  private loadConnectorDirectoriesUrl =this.BASE_URL+"loadconnectordirs";

  httpOptions = {
    headers: new HttpHeaders({
      //'Content-Type': 'application/json',
      'Access-Control-Allow-Origin':'*',      
  })
  };

  constructor(private http: HttpClient) { }

  hasLoginToken():Boolean{
    return localStorage.getItem('token') ? true : false;
  }

  saveLoginToken(data:any[]){
    // store some data in local storage (webbrowser)
    localStorage.clear();
    for(let i in data){
      localStorage.setItem(i ,data[i] );
    }
    //localStorage.setItem('usermeta' , JSON.stringify(this.userData));    
  }

  saveCurrentUserData(data:any){
    localStorage.setItem('userdata' , JSON.stringify(data));
  }

  saveItemToStorage(key:any,val:any){
    localStorage.setItem(key,val);
  }

  getItemFromStorage(key:any){
    return localStorage.getItem(key);
  }

  getCurrentUserData(){
    return JSON.parse(localStorage.getItem('userdata')||"{}") ;
  }

  getCurrentUserId(){
    return localStorage.getItem('id') ? localStorage.getItem('id') :'-1' ;
  }

  generateUserApiKey(postData:any){
    return this.http.post<any>(this.generateUserApiKeyDataUrl,postData , this.httpOptions);        
  }

  loadUserApiKey(postData:any){
    return this.http.post<any>(this.loadUserApiKeyDataUrl,postData , this.httpOptions);        
  }

  getCurrentSelectedApp(){
    let resp = undefined;
    let userData = localStorage.getItem('userdata')? localStorage.getItem('userdata') : 'no-app';
    if(userData){
      resp = JSON.parse(userData).app ;
    }
    return resp ? resp :'no-app';
  }

  getDefaultPage(){
    return localStorage.getItem('defaultPage') ;
  }

  getUserDataFromAPI(postData:any){
    return this.http.post<any>(this.userProfileDataUrl,postData , this.httpOptions);        
  }

  saveUserInfoData(postData:any){
    return this.http.post<any>(this.saveuserProfileUrl,postData , this.httpOptions);        
  }

  validateUserLogin(postData:any) :Observable<any>{
    return this.http.post<any>(this.userLoginUrl,postData , this.httpOptions);        
  }

  getMenuItemOptions(postData:any) : Observable<any>{
    return this.http.post<any>(this.getMenuItemsUrl,postData , this.httpOptions);        
  }

  getHomePageApps(postData:any) : Observable<any>{
    return this.http.post<any>(this.getHomePageItemsUrl,postData , this.httpOptions);        
  }

  getAppAlertItems(postData:any) : Observable<any>{
    return this.http.post<any>(this.getAppAlertItemsUrl,postData , this.httpOptions);        
  }

  getAppUserMsgItems(postData:any) : Observable<any>{
    return this.http.post<any>(this.getUserMsgItemsUrl,postData , this.httpOptions);        
  }

  getSavedFiles(fileName: String ):string{
    return this.getSavedFileDataUrl+"/"+fileName;
  }

  getS3BucketUrl(postData: any ):Observable<any>{
    return this.http.post<any>(this.getS3FileDataUrl, postData);
  }

  getAllSavedFiles(fileName: String ):string{
    return this.getAllSavedFileDataUrl+"/"+fileName;
  }

  runPrediction(selectedModel:string,postData:any){
    let url = "";
    console.log('predicting for',selectedModel);

    if(selectedModel=="Metal Casting Defects"){
      url = this.metalCastDefectDetectionDataUrl;
    }else if(selectedModel=="Surface Defects"){
      url=this.surfaceDefectDetectionDataUrl;
    }else if(selectedModel=="Hard Hat Present"){
      url=this.hardHatDetectionDataUrl;
    }else if(selectedModel=="Steel Defects"){
      url=this.steelDefectDetectionDataUrl;
    }else if(selectedModel=="Packaging Inspection"){
      url=this.packageDamageDetectionDataUrl;
    }
    else {
      url=this.inferenceForUserModelUrl;
    }
    return this.http.post<any>(url,postData , this.httpOptions);        
  }

  augmentImages(postData:any){
    return this.http.post<any>(this.imageAugmentationDataUrl,postData , this.httpOptions);        
  }

  createnewProject(postData:any){
    return this.http.post<any>(this.createProjectDataUrl,postData , this.httpOptions);        
  }

  loadAllProjects(){
    return this.http.get<any>(this.getProjectDataUrl , this.httpOptions);        
  }

  loadProjectScreenMeta(){
    return this.http.get<any>(this.getProjectMetaDataUrl , this.httpOptions);        
  }

  saveTrainTagsforProject(postData:any){
    return this.http.post<any>(this.saveTrainTagsforProjectUrl,postData , this.httpOptions);        
  }

  runTrainingForProject(postData:any){
    return this.http.post<any>(this.runTrainingForProjectUrl,postData , this.httpOptions);        
  }

  runInferencingForProject(postData:any){
    return this.http.post<any>(this.runInferencingForProjectUrl,postData , this.httpOptions);        
  }

  deleteProjectImages(postData:any){
    return this.http.post<any>(this.deleteProjectImagesUrl,postData , this.httpOptions);        
  }
  
  deleteProjectIterations(postData:any){
    return this.http.post<any>(this.deleteProjectItersUrl,postData , this.httpOptions);        
  }

  deleteProject(postData:any){
    return this.http.post<any>(this.deleteProjectUrl,postData , this.httpOptions);        
  }

  exportSampledImages(postData:any){
    return this.http.get<any>(this.getProjectMetaDataUrl , this.httpOptions);        
  }

  loadProjectStepsData(postData:any){
    return this.http.post<any>(this.loadProjectStepsDataUrl,postData , this.httpOptions);        
  }

  saveFilesToProjectData(postData:any){
    return this.http.post<any>(this.saveProjectStepDataUrl,postData , this.httpOptions);        
  }

  saveFilesToPredictionData(postData:any){
    return this.http.post<any>(this.savePredictionFilesDataUrl,postData , this.httpOptions);        
  }

  saveFilesToCustomLabelData(postData:any){
    return this.http.post<any>(this.saveCustomLabelDataUrl,postData , this.httpOptions);        
  }  

  saveFilesToAugData(postData:any){
    return this.http.post<any>(this.saveAugFilesDataUrl,postData , this.httpOptions);        
  }  

  saveFilesDataForCustomTrain(postData:any){
    return this.http.post<any>(this.saveFilesDataForCustomTrainUrl,postData , this.httpOptions);        
  }

  loadDraftCustomTrainData(postData:any){
    return this.http.post<any>(this.loadDraftCustomTrainDataUrl,postData , this.httpOptions);        
  }

  loadDraftCustomLabelData(postData:any){
    return this.http.post<any>(this.loadDraftCustomLabelDataUrl,postData , this.httpOptions);        
  }

  loadDraftPredictionData(postData:any){
    return this.http.post<any>(this.loadDraftPredictionDataUrl,postData , this.httpOptions);        
  }

  loadCustomLabelFilesData(postData:any){
    return this.http.post<any>(this.loadCustomLabelFileDataUrl,postData , this.httpOptions);        
  }

  loadAugmentationFilesData(postData:any){
    return this.http.post<any>(this.loadAugmentationFilesDataUrl,postData , this.httpOptions);        
  }

  loadAugmentationResources(postData:any){
    return this.http.post<any>(this.loadAllAugResourcesUrl,postData , this.httpOptions);        
  }

  saveAugmentationResLabel(postData:any){
    return this.http.post<any>(this.saveAugmentationResLabelUrl,postData , this.httpOptions);        
  }
  
  loadAllResources(postData:any){
    return this.http.post<any>(this.loadAllResourcesUrl,postData , this.httpOptions);        
  }

  runCustomTraining(postData:any){
    return this.http.post<any>(this.runCustomTrainingUrl,postData , this.httpOptions);        
  }

  getAllAvailableModels(postData:any){
    return this.http.post<any>(this.getAllAvailableModelDataUrl,postData , this.httpOptions);        
  }

  loadPredictionHistory(postData:any){
    return this.http.post<any>(this.loadPredictionHistoryDataUrl,postData , this.httpOptions);        
  }

  removePredictionHistory(postData:any){
    return this.http.post<any>(this.removePredictionHistoryDataUrl,postData , this.httpOptions);        
  }

  loadIterationHistory(postData:any){
    return this.http.post<any>(this.loadIterationsDataUrl,postData , this.httpOptions);        
  }

  loadAllProcessVideos(postData:any){
    return this.http.post<any>(this.loadAllProcessVideosDataUrl,postData , this.httpOptions);        
  }

  saveLabelledImages(postData:any){
    return this.http.post<any>(this.saveCustomLabelsForImgUrl,postData , this.httpOptions);        
  }

  getAllDataConnectors(postData:any){
    return this.http.post<any>(this.getAllDataConnectorsUrl,postData , this.httpOptions);        
  }

  loadConnectorDirectories(postData:any){
    return this.http.post<any>(this.loadConnectorDirectoriesUrl,postData , this.httpOptions);        
  }
}
