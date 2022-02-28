import { Component, HostListener, OnInit } from '@angular/core';
import { RestApiService } from 'src/app/services/restAPI/rest-api.service';
import * as JSZip from 'jszip';
import * as FileSaver from 'file-saver';
   

export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated: number
}

@Component({
  selector: 'app-data-sampling',
  templateUrl: './data-sampling.component.html',
  styleUrls: ['./data-sampling.component.scss']
})
export class DataSamplingComponent implements OnInit {

  dragAreaClass: string = "";
  loadedImgUrl: string = "";
  fileList : uploadFile[] = [];

  trainCount:number = 0;
  trainImgs: uploadFile[]=[];
  testCount:number = 0;
  testImgs: uploadFile[]=[];
  validationCount:number = 0;
  validationImgs: uploadFile[]=[];
  
  trainPercent:number = 0;
  testPercent:number = 0;
  validationPercent:number = 0;
  processing:boolean = false;

  modalPopUpId="folderImagePreview";

  constructor(private restAPIService : RestApiService) { }

  ngOnInit(): void {
    this.dragAreaClass = "dragarea";
    this.setDefaultSettingsValue();
  }

  setDefaultSettingsValue(){
    this.trainPercent=70;
    this.testPercent=20;
    this.validationPercent=10;
  }

  samplingImages(data:FormData){
    console.log('posed form',data);
    // randomising the images 
    this.fileList.sort((a, b) => Math.random() - 0.5);

    this.trainCount=Math.floor((this.trainPercent/100)*(this.fileList.length));
    this.testCount=Math.floor((this.testPercent/100)*(this.fileList.length));
    this.validationCount= this.fileList.length - (this.trainCount+this.testCount);

    this.trainImgs=this.fileList.splice(0,this.trainCount);
    this.testImgs=this.fileList.splice(0,this.testCount);
    this.validationImgs=this.fileList.splice(0,this.validationCount);
    
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
    }
  }

  saveFiles(files: FileList) {

    this.fileList = [];
    for(let i=0; i < files.length; i++){

      this.fileList.push({'no':i,'file':files[i],'name':files[i].name,'path':'','updated':files[i].lastModified});        
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


onselectionChange(event:Event,change: any){
  event.preventDefault();
  console.log('e',change,event);
  let tmp = this.fileList.filter(f=>f.no==change);
  let reader = new FileReader();
  reader.onload = (e: any) => {
    this.loadedImgUrl=e.target.result;
  }
  if (tmp && tmp.length>0){
    reader.readAsDataURL(tmp[0].file);  
  }

}

  onFileChange(event:any){
    console.log('event fired ->',event)
  }

  exportSampledImages(event:any){
    this.processing = true;
    event.preventDefault();
    let zip = new JSZip();
    zip.file("visimatic_info.txt", 'This folder contains the split file into corresponding folders');
    let train = zip.folder("train");  
    let test = zip.folder("test");   
    let validation = zip.folder("validation");   

    if (train) {
      for(let i =0; i < this.trainImgs.length; i++){
        train.file(this.trainImgs[i].name,this.trainImgs[i].file);
      }
    }

    if (test) {
      for(let i =0; i < this.testImgs.length; i++){
        test.file(this.testImgs[i].name,this.testImgs[i].file);
      }
    }
    
    if (validation) {
      for(let i =0; i < this.validationImgs.length; i++){
        validation.file(this.validationImgs[i].name,this.validationImgs[i].file);
      }
    }

    zip.generateAsync({ type: "blob" })
    .then((content)=> {
      FileSaver.saveAs(content, "Visimatic_Export.zip");
      this.processing = false;
    });
  }

}
