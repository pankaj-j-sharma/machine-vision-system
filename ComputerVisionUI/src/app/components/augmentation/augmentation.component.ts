import { Component, HostListener, OnInit } from '@angular/core';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';

declare var jQuery:any;

export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated?: number
  augImgs: AugImgFile[]
  resId : number  
}

export interface AugImgFile {
  no:number
  name:string
  url:string
  imgClass:string
}

@Component({
  selector: 'app-augmentation',
  templateUrl: './augmentation.component.html',
  styleUrls: ['./augmentation.component.scss']
})
export class AugmentationComponent implements OnInit {

  error: string= "";
  dragAreaClass: string = "";
  loadedImgUrl: string = "";
  placeholderImg= "assets/images/placeholder.jpg";
  fileList : uploadFile[] = [];
  fileListAug : uploadFile[] = [];
  augmentedFiles: AugImgFile[] = [];

  rescalingOption = "None Selected";
  flipOption = "None Selected";
  translationOption = "None Selected";
  rotationOption = "None Selected";
  shearingOption = "Yes/No";
  formData = new FormData();
  processing:boolean = false;
  arrayOfBlob = new Array<Blob>();
  emptyFile:File = new File(this.arrayOfBlob, "example.zip", { type: 'application/zip' })
  allResources: any[] = [];  
  saveAsLabel : string = "";
  augId:number = -1;
  resourceId:number = -1;
  userId = this.restAPIService.getCurrentUserId();
  defaultSelection:number = -1;

  constructor(private restAPIService : RestApiService) { }

  ngOnInit(): void {
    this.restAPIService.saveItemToStorage('current',window.location.pathname);
    this.dragAreaClass = "dragarea";
    this.loadAllResources();
  }  

  @HostListener("dragover", ["$event"]) onDragOver(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragenter", ["$event"]) onDragEnter(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragend", ["$event"]) onDragEnd(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("dragleave", ["$event"]) onDragLeave(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("drop", ["$event"]) onDrop(event: any) {
    this.dragAreaClass = "dragarea";
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer.files) {
      let files: FileList = event.dataTransfer.files;
      //console.log('files',files)
      this.saveFiles(files);
      this.saveFilesToAugHistoryData();
    }
  }

  saveFiles(files: FileList) {
    this.processing=true;
    this.fileList = [];
    for(let i=0; i < files.length; i++){

      this.fileList.push({'no':i,'file':files[i],'name':files[i].name,'path':'','updated':files[i].lastModified ,resId:-1,augImgs:[]});        
      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = ((theFile) => {
          return (e:any) => {
            this.fileList.filter(f=>f.name==theFile.name)[0].path=e.target.result;              
          };
      })(files[i]);

      // Read in the image file as a data URL.
      reader.readAsDataURL(files[i]);        
    }
  }


saveFilesToAugHistoryData(){
  this.processing = true;
  this.formData = new FormData();
  this.formData.append('name','Custom Augmentation');
  this.formData.append('desc','Custom Augmentation');
  this.formData.append('augId',this.augId.toString());
  this.formData.append('resourceId',this.resourceId.toString());
  let userId = this.restAPIService.getCurrentUserId();
  if (userId){
    this.formData.append('userId',userId.toString());
  }    
  this.fileList.forEach(f=> {
    this.formData.append(f.name,f.file);
  });

  this.restAPIService.saveFilesToAugData(this.formData).subscribe(resp => {
    console.log('saveFilesToAugmentationData',resp);
    if('success' in resp){
      let results = resp.results;
      let resultFiles = results.files;
      this.augId = results.augId;
      this.resourceId = results.resId;
      this.fileListAug = [];
      for (let i = 0; i < resultFiles.length; i++) {
        this.fileListAug.push({'path':this.restAPIService.getAllSavedFiles(resultFiles[i].path),no:resultFiles[i].no,file:this.emptyFile,updated:resultFiles[i].updated,name:resultFiles[i].name,augImgs:[],resId:resultFiles[i].resId});

        // load first image by default 
        if(i==0){
          this.loadedImgUrl = this.restAPIService.getAllSavedFiles(resultFiles[i].path);
          this.augmentedFiles = []; 
          this.defaultSelection = i; 
        }      
        
      }
    }
    this.processing = false;
  });

}

augmentImage(data:any){
  this.formData = new FormData();
  console.log('data->',data,'augId',this.augId.toString());
  Object.keys(data).forEach(k=>{
    this.formData.append(k,data[k]); 
  });
  this.formData.append('augId',this.augId.toString());
  this.formData.append('resourceId',this.resourceId.toString());
  if (this.userId){
    this.formData.append('userId',this.userId.toString());
  }    
  this.processing=true;
  this.restAPIService.augmentImages(this.formData).subscribe(resp=>{
    this.fileList=[];
    this.fileListAug=[];
    this.processing=false;
    let results = resp.results;
    for (let i = 0; i < results.length; i++) {
      let tmpFile:uploadFile = {no:i,file:this.emptyFile,name:results[i].name,path:this.restAPIService.getAllSavedFiles(results[i].path),updated:undefined,augImgs:[],resId:-1};
      tmpFile.augImgs?.push({no:1,name:'Resize',url:results[i].resize==undefined?this.placeholderImg:this.restAPIService.getAllSavedFiles(results[i].resize),imgClass:'m-augImg'});
      tmpFile.augImgs?.push({no:2,name:'Flip',url:results[i].flip==undefined?this.placeholderImg: this.restAPIService.getAllSavedFiles(results[i].flip),imgClass:'m-augImg'});
      tmpFile.augImgs?.push({no:3,name:'Translate',url:results[i].translate==undefined?this.placeholderImg: this.restAPIService.getAllSavedFiles(results[i].translate),imgClass:'m-augImg'});
      tmpFile.augImgs?.push({no:4,name:'Rotate',url:results[i].rotate==undefined?this.placeholderImg: this.restAPIService.getAllSavedFiles(results[i].rotate),imgClass:'m-augImg'});
      tmpFile.augImgs?.push({no:5,name:'Shear',url:results[i].shear==undefined?this.placeholderImg: this.restAPIService.getAllSavedFiles(results[i].shear),imgClass:'m-augImg'});
      this.fileListAug.push(tmpFile);

      // load first image by default 
      if(i==0){
        this.loadedImgUrl = this.restAPIService.getAllSavedFiles(results[i].path);
        if(tmpFile.augImgs){
          this.augmentedFiles = tmpFile.augImgs; 
        }
        this.defaultSelection = i; 
      }      
    }
    console.log('results ->',results,'files ->',this.fileListAug);
  });      

}

onselectionChange(event:Event,change: any){
  this.defaultSelection = -1;
  event.preventDefault();
  this.loadMetaData(); // for dummy sample   
  console.log('e',change,event);
  let tmp = this.fileListAug.filter(f=>f.no==change);
  if(tmp && tmp.length>0){
    this.loadedImgUrl = tmp[0].path;
    this.augmentedFiles = tmp[0].augImgs;
  }
}

selectPreviewOptn(no:number){
  
  this.augmentedFiles.forEach((file)=>{
    if(file.no==no){
      file.imgClass+=" prvwImgSelected";
      this.loadedImgUrl = file.url;
    }
    else {
      file.imgClass="m-augImg";
    }
  });
}

  onFileChange(event:any){
    console.log('event fired ->',event)
  }


  saveResults(save:Boolean){
    console.log(save?"Save":"Undo","the results");
  }

  removeUpload(){
    this.fileList=[];
    this.fileListAug=[];
    this.loadedImgUrl=this.placeholderImg;
    this.augId=-1;
    this.resourceId=-1;
    this.loadMetaData();
  }  
  
  loadMetaData(){
    this.augmentedFiles=[
      {
        no:1,
        name:'rescaled',
        url:this.placeholderImg,
        imgClass:'m-augImg'
      },
      {
        no:2,
        name:'flip',
        url:this.placeholderImg,
        imgClass:'m-augImg'
      },
      {
        no:3,
        name:'translation',
        url:this.placeholderImg,
        imgClass:'m-augImg'
      },
      {
        no:4,
        name:'rotation',
        url:this.placeholderImg,
        imgClass:'m-augImg'
      },
      {
        no:5,
        name:'shearing',
        url:this.placeholderImg,
        imgClass:'m-augImg'
      },
    ];

  }

  loadAllResources(){
    this.loadMetaData();
    this.formData = new FormData();
    this.formData.append('screen','Augmentation');
    this.restAPIService.loadAugmentationResources(this.formData).subscribe(resp =>{
      if('success' in resp){
        let results = resp.results;
        this.allResources = results.all_resources;
      }
      this.processing = false;
    });
  }

  loadSaveAsModal(){
    jQuery(".bd-example-modal-sm").modal("show");
  }

  saveAs(data:any){
    this.saveAsLabel = data.saveAsLabel;
    jQuery(".bd-example-modal-sm").modal("hide");
    this.saveAugmentationResults();
  }

  saveAugmentationResults(){
    console.log('res',this.resourceId,this.augId,this.saveAsLabel);
    this.processing=true;
    this.formData = new FormData();
    this.formData.append('screen','Augmentation');
    this.formData.append('augId',this.augId.toString());
    if (this.userId){
      this.formData.append('userId',this.userId.toString());
    }    
    this.formData.append('desc',this.saveAsLabel.toString());
    this.formData.append('resourceId',this.resourceId.toString());
    this.restAPIService.saveAugmentationResLabel(this.formData).subscribe(resp =>{
      if('success' in resp){
        console.log('resp ',resp);
        this.loadAllResources();
      }
      this.processing= false;
    });
  }

  selectImageResource(event:any){
    this.processing = true;

    // clear existing images and augmented files 
    this.removeUpload();

    // dont load anything if default option selected 
    if(!event.target.value){
      this.processing=false;
      return;
    }
    this.resourceId = event.target.value;
    this.fileListAug = [];    
    this.formData = new FormData();
    this.formData.append('augId',this.augId==undefined?"-1":this.augId.toString());
    if (this.userId){
      this.formData.append('userId',this.userId.toString());
    }    
    this.formData.append('resourceId',this.resourceId.toString());
    this.restAPIService.loadAugmentationFilesData(this.formData).subscribe(resp1 => {
      console.log('resp1',resp1);
      if ('success' in resp1){
        let results1 = resp1.results;
        this.augId = resp1.augId;
        for (let i = 0; i < results1.length; i++) {
          let imgAugs = results1[i].augImgs;
          console.log('imgAugs1',imgAugs);
          for (let j = 0; j < imgAugs.length; j++) {
            imgAugs[j].no = j;
            imgAugs[j].url = imgAugs[j].url==""?this.placeholderImg:this.restAPIService.getAllSavedFiles(imgAugs[j].url);
            imgAugs[j].imgClass = 'm-augImg';            
          }
          console.log('imgAugs2',imgAugs);
          this.fileListAug.push({'path':this.restAPIService.getAllSavedFiles(results1[i].path),no:i,augImgs:imgAugs,file:this.emptyFile,updated:1,name:results1[i].name,resId:results1[i].id});

          // load first image by default 
          if(i==0){
            this.loadedImgUrl = this.restAPIService.getAllSavedFiles(results1[i].path);
            if(imgAugs){
              this.augmentedFiles = imgAugs; 
            }
            this.defaultSelection = i; 
          }
    
        }
      }
      this.processing = false;
  
    });

  }  

}
