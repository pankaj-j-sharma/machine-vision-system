import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataConnectorsComponent } from './data-connectors.component';

describe('DataConnectorsComponent', () => {
  let component: DataConnectorsComponent;
  let fixture: ComponentFixture<DataConnectorsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DataConnectorsComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DataConnectorsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
