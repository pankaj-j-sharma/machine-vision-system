import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataSamplingComponent } from './data-sampling.component';

describe('DataSamplingComponent', () => {
  let component: DataSamplingComponent;
  let fixture: ComponentFixture<DataSamplingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DataSamplingComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DataSamplingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
