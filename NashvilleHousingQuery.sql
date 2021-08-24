
SELECT * 
FROM PortfolioProject..NashvilleHousing


SELECT SaleDateConverted 
FROM PortfolioProject..NashvilleHousing

UPDATE NashvilleHousing
SET SaleDate = CONVERT(Date,SaleDate)

ALTER TABLE NashvilleHousing
ADD SaleDateConverted Date; 

UPDATE NashvilleHousing
SET SaleDateConverted = CONVERT(Date,SaleDate)



----------
--populate propery address data 

SELECT *
FROM PortfolioProject..NashvilleHousing
WHERE PropertyAddress IS NULL


--if parcel id has an address and another parcel 
-- doesn't have, let's populate it with the same address 

SELECT *
FROM PortfolioProject..NashvilleHousing
ORDER BY ParcelID

SELECT a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM PortfolioProject..NashvilleHousing as a 
--let's do a self join 
JOIN PortfolioProject..NashvilleHousing as b 
ON a.ParcelID = b.ParcelID 
AND a.[UniqueID ]<> b.[UniqueID]
WHERE a.PropertyAddress IS NULL

UPDATE a
SET PropertyAddress = ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM PortfolioProject..NashvilleHousing as a 
JOIN PortfolioProject..NashvilleHousing as b 
ON a.ParcelID = b.ParcelID 
AND a.[UniqueID ]<> b.[UniqueID]


----------------------
---Breaking out Address into Individual Columns (Address, City, State) 


SELECT PropertyAddress 
FROM PortfolioProject..NashvilleHousing


SELECT 
SUBSTRING(PropertyAddress, 1, CHARINDEX(',' , PropertyAddress) -1) as Address 
, SUBSTRING(PropertyAddress,CHARINDEX(',' , PropertyAddress) + 1, LEN(PropertyAddress )) as Address 
FROM PortfolioProject..NashvilleHousing


ALTER TABLE NashvilleHousing
ADD PropertySplitAddress Nvarchar(255); 

UPDATE NashvilleHousing
SET PropertySplitAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX(',' , PropertyAddress) -1)

ALTER TABLE NashvilleHousing
ADD PropertySplitCity Nvarchar(255); 


UPDATE NashvilleHousing
SET PropertySplitCity =  SUBSTRING(PropertyAddress,CHARINDEX(',' , PropertyAddress) + 1, LEN(PropertyAddress )) 



SELECT *
FROM PortfolioProject.dbo.NashvilleHousing



SELECT OwnerAddress
FROM PortfolioProject.dbo.NashvilleHousing

--parse name 

SELECT
PARSENAME(REPLACE(OwnerAddress, ',', '.') , 3)
,PARSENAME(REPLACE(OwnerAddress, ',', '.') , 2)
,PARSENAME(REPLACE(OwnerAddress, ',', '.') , 1)
FROM PortfolioProject..NashvilleHousing


ALTER TABLE NashvilleHousing
ADD OwnerSplitAddress Nvarchar(255); 

UPDATE NashvilleHousing
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress, ',', '.') , 3)

ALTER TABLE NashvilleHousing
ADD OwnerSplitCity Nvarchar(255); 

UPDATE NashvilleHousing
SET OwnerSplitCity =  PARSENAME(REPLACE(OwnerAddress, ',', '.') , 2)

ALTER TABLE NashvilleHousing
ADD OwnerSplitState Nvarchar(255); 

UPDATE NashvilleHousing
SET OwnerSplitState =  PARSENAME(REPLACE(OwnerAddress, ',', '.') , 1)


---------------------
--Change Y and N to Yes and No in "Sold as Vacant" fields 

SELECT Distinct(SoldAsVacant), COUNT(SoldAsVacant) 
FROM PortfolioProject..NashvilleHousing
GROUP BY SoldAsVacant
ORDER BY 2 


SELECT SoldAsVacant
, CASE WHEN SoldAsVacant = 'Y' THEN  'Yes'
	   WHEN SoldAsVacant = 'N' THEN  'No'
	   ELSE SoldAsVacant 
	   END
	
FROM PortfolioProject..NashvilleHousing

UPDATE NashvilleHousing
SET SoldAsVacant = CASE WHEN SoldAsVacant = 'Y' THEN  'Yes'
	   WHEN SoldAsVacant = 'N' THEN  'No'
	   ELSE SoldAsVacant 
	   END
FROM PortfolioProject..NashvilleHousing

SELECT Distinct(SoldAsVacant), COUNT(SoldAsVacant) 
FROM PortfolioProject..NashvilleHousing
GROUP BY SoldAsVacant
ORDER BY 2 



------------------------------
--Remove Duplicates


WITH RowNumCTE AS(
SELECT * , 
ROW_NUMBER() OVER( 
PARTITION BY ParcelID,
			PropertyAddress,
			SalePrice,
			SaleDate,
			LegalReference
			ORDER BY 
				UniqueID
				) row_num
FROM PortfolioProject..NashvilleHousing
)
DELETE
FROM RowNumCTE
WHERE row_num > 1 




------------------------------------
--Delete Unused Columns


SELECT * 
FROM PortfolioProject.dbo.NashvilleHousing

ALTER TABLE PortfolioProject..NashvilleHousing
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress

ALTER TABLE PortfolioProject..NashvilleHousing
DROP COLUMN SaleDate



SELECT * 
FROM PortfolioProject.dbo.NashvilleHousing











------------------------