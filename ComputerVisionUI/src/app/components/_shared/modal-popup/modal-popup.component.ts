import { Component, Input, OnInit } from '@angular/core';

export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated: number
}

@Component({
  selector: 'app-modal-popup',
  templateUrl: './modal-popup.component.html',
  styleUrls: ['./modal-popup.component.scss']
})
export class ModalPopupComponent implements OnInit {

  @Input() modalId :string=""; 
  @Input() imgFileList : uploadFile[] = [];
  arrOfRows:number[]=[];
  numberOfCols:number=9;
  

  constructor() { }

  ngOnInit(): void {
    this.loadMetaData();
  }

  loadMetaData(){
    //console.log('passed file list ',this.imgFileList);
    this.arrOfRows=Array.from({length:Math.ceil(this.imgFileList.length/this.numberOfCols)+1},(x,i) => i );
  }

}
